from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from utils.box_decoder import loc2bbox
from torchvision.ops import nms

def normal_init(m, mean, stddev):
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()


class ProposalGenerator():
    def __init__(self,
                 mode,
                 nms_iou=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=600,
                 n_test_pre_nms=3000,
                 n_test_post_nms=300,
                 min_size=16
                 ):

        self.mode = mode
        self.nms_iou = nms_iou

        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms

        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        anchor = torch.from_numpy(anchor).type_as(loc)
        roi = loc2bbox(anchor, loc)

        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])

        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]

        roi = roi[keep, :]
        score = score[keep]

        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        keep = nms(roi, score, self.nms_iou)

        if len(keep) < n_post_nms:
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep = torch.cat([keep, keep[index_extra]])
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi


class RegionProposalNetwork(nn.Module):
    def __init__(
            self,
            in_channels=1024,
            mid_channels=1024,
            ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32],
            feat_stride=16,
            mode="training",
    ):
        super(RegionProposalNetwork, self).__init__()
        self.base_anchors = self.generate_anchors(anchor_scales, ratios)
        n_anchor = self.base_anchors.shape[0]

        self.feat_stride = feat_stride
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        self.proposal_generator = ProposalGenerator(mode, min_size=feat_stride)


        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def generate_anchors(self, scales, ratios, base_size=16):
        base_anchors = np.zeros((len(scales) * len(ratios), 4), dtype=np.float32)
        ind = 0
        for s in scales:
            for r in ratios:
                h = base_size * s * np.sqrt(r)
                w = base_size * s * np.sqrt(1. / r)

                base_anchors[ind, :] = np.array([-h/2, -w/2, h/2, w/2]).reshape(1, 4)
                ind += 1
        return base_anchors

    def shift_anchors(self, feat_stride, base_anchors, height, width):
        shift_x = np.arange(0, width * feat_stride, feat_stride)
        shift_y = np.arange(0, height * feat_stride, feat_stride)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

        A = base_anchors.shape[0]
        K = shift.shape[0]
        anchor = base_anchors.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
        anchor = anchor.reshape((K * A, 4)).astype(np.float32)
        return anchor

    def forward(self, x, img_size, scale=1.0):
        n, c, h, w = x.shape
        x = F.relu(self.conv1(x))
        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)

        anchors = self.shift_anchors(self.feat_stride, self.base_anchors, h, w)

        rois = list()
        roi_indices = list()

        for i in range(n):
            roi = self.proposal_generator(rpn_locs[i], rpn_fg_scores[i], anchors, img_size, scale=scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_indices.append(batch_index.unsqueeze(0))

        rois = torch.cat(rois, dim=0).type_as(x)
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)
        anchors = torch.from_numpy(anchors).unsqueeze(0).float().to(x.device)

        return rpn_locs, rpn_scores, rois, roi_indices, anchors







