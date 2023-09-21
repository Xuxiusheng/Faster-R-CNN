import torch
from models.backbone import resnet50
from models.rpn import RegionProposalNetwork
from models.RoIHead import Resnet50RoIHead
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from utils.box_decoder import DecodeBox
from utils.utils import get_new_img_size, resize_image
from utils.utils_map import get_map
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore")

def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc

class AnchorGenerator(object):
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def calc_ious(self, anchor, bbox):
        ious = bbox_iou(anchor, bbox)
        if len(bbox) == 0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        argmax_ious = ious.argmax(axis=1)
        max_ious = np.max(ious, axis=1)
        gt_argmax_ious = ious.argmax(axis=0)
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious

    def create_label(self, anchor, bbox):
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)
        argmax_ious, max_ious, gt_argmax_ious = self.calc_ious(anchor, bbox)

        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        if len(gt_argmax_ious) > 0:
            label[gt_argmax_ious] = 1

        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def __call__(self, bbox, anchor):
        argmax_ious, label = self.create_label(anchor, bbox)
        if (label > 0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label

class Proposal(object):
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        iou = bbox_iou(roi, bbox)

        if len(bbox) == 0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            gt_assignment = iou.argmax(axis=1)
            max_iou = iou.max(axis=1)
            gt_roi_label = label[gt_assignment] + 1

        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        keep_index = np.append(pos_index, neg_index)
        sample_roi = roi[keep_index]

        if len(bbox) == 0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        return sample_roi, gt_roi_loc, gt_roi_label



class FRCNNTrainer(object):
    def __init__(self, dataloader, device, opt, num_classes, class_names):

        self.opt = opt
        self.device = device
        self.dataloader = dataloader
        self.epochs = opt.epochs
        self.rpn_sigma = 1.0
        self.roi_sigma = 1.0
        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]
        self.epoch = [0]
        self.map = [0]
        self.best_map = 0
        self.nms_iou = opt.nms_iou
        self.confidence = opt.confidence
        self.max_boxes = opt.max_boxes
        self.map_out_path = opt.map_out_path
        self.class_names = class_names
        self.MINOVERLAP = opt.MINOVERLAP
        if not os.path.exists(self.map_out_path):
            os.mkdir(self.map_out_path)

        if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
            os.mkdir(os.path.join(self.map_out_path, "detection-results"))

        if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
            os.mkdir(os.path.join(self.map_out_path, "ground-truth"))

        self.extractor, self.classifier = resnet50(opt.pretrain_backbone)
        self.extractor = self.extractor.to(self.device)

        self.rpn = RegionProposalNetwork(
            1024, 512,
            ratios=opt.ratios,
            anchor_scales=opt.scales,
            feat_stride=16,
            mode="training"
        ).to(device)

        self.anchor_generator = AnchorGenerator()
        self.proposal = Proposal()
        self.head = Resnet50RoIHead(
            n_class=num_classes + 1,
            roi_size=38,
            spatial_scale=1,
            classifier=self.classifier
        ).to(device)

        self.optimizer = optim.Adam([{"params": self.extractor.parameters()},
                                     {"params": self.rpn.parameters()},
                                     {"params": self.head.parameters()}],
                                    lr=self.opt.lr)
        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(num_classes + 1)[None]
        self.decoder = DecodeBox(self.std.to(self.device), num_classes)
        self.flag = False
        if opt.freeze_backbone:
            for name, param in self.extractor.named_parameters():
                param.requires_grad_(False)
            self.flag = True
            print("backbone parameters have been frozen!")

    def fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        pred_loc = pred_loc[gt_label > 0]
        gt_loc = gt_loc[gt_label > 0]

        sigma_squared = sigma ** 2
        regression_diff = (gt_loc - pred_loc)
        regression_diff = regression_diff.abs().float()
        regression_loss = torch.where(
            regression_diff < (1. / sigma_squared),
            0.5 * sigma_squared * regression_diff ** 2,
            regression_diff - 0.5 / sigma_squared
        )
        regression_loss = regression_loss.sum()
        num_pos = (gt_label > 0).sum().float()

        regression_loss /= torch.max(num_pos, torch.ones_like(num_pos))
        return regression_loss

    def train_step(self, epoch):
        if self.opt.freeze_backbone and self.flag and epoch >= self.opt.freeze_epoch:
            for name, param in self.extractor.named_parameters():
                param.requires_grad_(True)
            self.flag = False
            print("backbone parameters have been unfrozen!")

        self.extractor.train()
        self.rpn.train()
        self.head.train()
        with tqdm(total=len(self.dataloader), desc=f'Epoch {epoch + 1}/{self.epochs}', postfix=dict, mininterval=0.3) as pbar:
            for k, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                images, boxes, labels = batch[0], batch[1], batch[2]
                img_size = images.shape[-2:]
                n = images.shape[0]
                images = images.to(self.device)
                base_features = self.extractor(images)
                rpn_locs, rpn_scores, rois, roi_indices, anchors = self.rpn(base_features, img_size, self.opt.scale)
                rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = 0, 0, 0, 0
                sample_rois, sample_indexes, gt_roi_locs, gt_roi_labels = [], [], [], []

                for i in range(n):
                    bbox = boxes[i]
                    label = labels[i]
                    rpn_loc = rpn_locs[i]
                    rpn_score = rpn_scores[i]
                    roi = rois[i]
                    gt_rpn_loc, gt_rpn_label = self.anchor_generator(bbox, anchors[0].cpu().numpy())

                    gt_rpn_loc = torch.Tensor(gt_rpn_loc).type_as(rpn_locs)
                    gt_rpn_label = torch.Tensor(gt_rpn_label).type_as(rpn_locs).long()

                    rpn_loc_loss = self.fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
                    rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

                    rpn_loc_loss_all += rpn_loc_loss
                    rpn_cls_loss_all += rpn_cls_loss

                    sample_roi, gt_roi_loc, gt_roi_label = self.proposal(roi, bbox, label, self.loc_normalize_std)

                    sample_rois.append(torch.Tensor(sample_roi).type_as(rpn_locs))
                    sample_indexes.append(torch.ones(len(sample_roi)).type_as(rpn_locs) * roi_indices[i][0])
                    gt_roi_locs.append(torch.Tensor(gt_roi_loc).type_as(rpn_locs))
                    gt_roi_labels.append(torch.Tensor(gt_roi_label).type_as(rpn_locs).long())

                sample_rois = torch.stack(sample_rois, dim=0)
                sample_indexes = torch.stack(sample_indexes, dim=0)

                roi_cls_locs, roi_scores = self.head(base_features, sample_rois, sample_indexes, img_size, self.device)

                for i in range(n):
                    n_sample = roi_cls_locs.shape[1]
                    roi_cls_loc = roi_cls_locs[i]
                    roi_score = roi_scores[i]
                    gt_roi_loc = gt_roi_locs[i]
                    gt_roi_label = gt_roi_labels[i]
                    roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
                    roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]

                    roi_loc_loss = self.fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
                    roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

                    roi_loc_loss_all += roi_loc_loss
                    roi_cls_loss_all += roi_cls_loss

                losses = [rpn_loc_loss_all / n, rpn_cls_loss_all / n, roi_loc_loss_all / n, roi_cls_loss_all / n]
                losses = losses + [sum(losses)]

                losses[-1].backward()
                self.optimizer.step()

                pbar.set_postfix(**{"total_loss": losses[-1].item(),
                                    "rpn_loc_loss": losses[0].item(),
                                    "rpn_cls_loss": losses[1].item(),
                                    "roi_loc_loss": losses[2].item(),
                                    "roi_cls_loss": losses[3].item()})

                pbar.update(1)

    def generate_txt(self, valid_lines, epoch):
        print("Start generate result file....")
        self.extractor.eval()
        self.rpn.eval()
        self.head.eval()
        with tqdm(total=len(valid_lines), desc=f'Epoch {epoch + 1}/{self.epochs}', postfix=dict,
                  mininterval=0.3) as pbar:
            for annotation_line in valid_lines:
                line = annotation_line.split()
                image_id = os.path.basename(line[0]).split('.')[0]
                f = open(os.path.join(self.map_out_path, "detection-results/" + image_id + ".txt"), "w")
                image = Image.open(line[0])

                gt_boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
                if epoch == 0:
                    with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                        for box in gt_boxes:
                            left, top, right, bottom, obj = box
                            obj_name = self.class_names[obj]
                            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

                image_shape = np.array(np.shape(image)[0:2])
                input_shape = get_new_img_size(image_shape[0], image_shape[1])
                image_data = np.array(resize_image(image, [input_shape[1], input_shape[0]]), dtype=np.float32) / 255.0
                image_data = np.transpose(image_data, (2, 0, 1))[None]

                with torch.no_grad():
                    images = torch.from_numpy(image_data).to(self.device)
                    roi_cls_locs, roi_scores, rois, _ = self.predict(images)
                    result = self.decoder.forward(roi_cls_locs, roi_scores, rois,
                                                    image_shape, input_shape,
                                                    nms_iou=self.nms_iou, confidence=self.confidence)
                    if len(result[0]) <= 0:
                        f.close()
                        continue

                    top_label = np.array(result[0][:, 5], dtype='int32')
                    top_conf = result[0][:, 4]
                    top_boxes = result[0][:, :4]

                top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
                top_boxes = top_boxes[top_100]
                top_conf = top_conf[top_100]
                top_label = top_label[top_100]

                for i, c in list(enumerate(top_label)):
                    predicted_class = self.class_names[int(c)]
                    box = top_boxes[i]
                    score = str(top_conf[i])
                    top, left, bottom, right = box
                    if predicted_class not in self.class_names:
                        continue

                    f.write("%s %s %s %s %s %s\n" % (
                    predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
                f.close()


                pbar.set_postfix(**{"file": image_id})
                pbar.update(1)

        print("result files have been generated sucessfully")

    def calc_map(self):
        print("Start calculate mAP")
        mAP = get_map(self.MINOVERLAP, self.opt.draw_plot, path=self.map_out_path)
        print("mAP has been calculated")

        return mAP

    def predict(self, images):
        img_size = images.shape[-2:]
        images = images.to(self.device)
        base_feature = self.extractor(images)
        rpn_locs, rpn_scores, rois, roi_indices, anchors = self.rpn(base_feature, img_size, self.opt.scale)

        roi_cls_locs, roi_scores = self.head(base_feature, rois, roi_indices, img_size, self.device)

        return roi_cls_locs, roi_scores, rois, roi_indices


    def save(self, epoch, best=False):
        ckpt = {
            "backbone": self.extractor.state_dict(),
            "rpn": self.rpn.state_dict(),
            "head": self.head.state_dict(),
            "epoch": epoch,
            "best_map": self.best_map,
            "maps": self.map,
            "epochs": self.epoch
        }
        torch.save(ckpt, self.opt.save_dir + "/last.pth")
        if best:
            torch.save(ckpt, self.opt.save_dir + "/best.pth")

    def load(self, path):
        ckpt = torch.load(path)
        self.extractor.load_state_dict(ckpt["backbone"])
        self.rpn.load_state_dict(ckpt["rpn"])
        self.head.load_state_dict(ckpt["head"])
        self.epoch = ckpt["epochs"]
        self.map = ckpt["maps"]
        self.best_map = ckpt["best_map"]

        return ckpt["epoch"]