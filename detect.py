import torch
from PIL import Image, ImageDraw, ImageFont
from utils.utils import get_classes, get_new_img_size, resize_image
from utils.box_decoder import DecodeBox
import argparse
from models.backbone import resnet50
from models.rpn import RegionProposalNetwork
from models.RoIHead import Resnet50RoIHead
import numpy as np
import colorsys
import cv2
import os

class Detector(object):
    def __init__(self, opt):
        self.opt = opt
        self.device = "cpu" if torch.cuda.is_available() else "cpu"
        self.class_names, self.num_classes = get_classes(opt.classes_dir)
        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None].to(self.device)
        self.decoder = DecodeBox(self.std, self.num_classes)
        self.extractor, self.classifier = resnet50(False)
        self.extractor = self.extractor.to(self.device)
        self.rpn = RegionProposalNetwork(1024, 512, mode="testing").to(self.device)
        self.head = Resnet50RoIHead(
            n_class=self.num_classes + 1,
            roi_size=38,
            spatial_scale=1,
            classifier=self.classifier
        ).to(self.device)
        self.load_weights(opt)
        self.extractor.eval()
        self.rpn.eval()
        self.head.eval()

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def load_weights(self, opt):
        ckpt = torch.load(opt.model_dir)
        self.extractor.load_state_dict(ckpt["backbone"])
        self.rpn.load_state_dict(ckpt["rpn"])
        self.head.load_state_dict(ckpt["head"])


    def detect(self, image):
        image_shape = np.array(np.shape(image)[:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        image_data = np.array(resize_image(image, [input_shape[1], input_shape[0]]), dtype=np.float32) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))[None]
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            roi_cls_locs, roi_scores, rois, _ = self.predict(images)
            results = self.decoder.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape,
                                           nms_iou=self.opt.nms_iou, confidence=self.opt.confidence)
            if len(results[0]) <= 0:
                print("no object in image")
                return image

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]
        print(f"detect {len(results[0])} object")

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])

            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)

            del draw

        return image

    def detect_image(self, path):
        image = Image.open(path)
        image = self.detect(image)
        if self.opt.show:
            image.show()

        npath = os.path.join(self.opt.save_dir, os.path.basename(path))
        image.save(npath)


    def detect_video(self, path):
        fps = 24
        npath = os.path.join(self.opt.save_dir, os.path.basename(path))
        flag = False

        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not flag:
                size = frame.shape[:2]
                video = cv2.VideoWriter(npath, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
                flag = True

            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = self.detect(image)

                img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                video.write(img)
                if self.opt.show:
                    cv2.imshow("Video", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        video.release()
        cv2.destroyAllWindows()

    def predict(self, x):
        img_size = x.shape[2:]
        base_feature = self.extractor.forward(x)
        _, _, rois, roi_indices, _ = self.rpn(base_feature, img_size, self.opt.scale)
        roi_cls_locs, roi_scores = self.head(base_feature, rois, roi_indices, img_size, self.device)
        return roi_cls_locs, roi_scores, rois, roi_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Faster-RCNN-Detector")
    parser.add_argument("--classes_dir", type=str, default="voc_classes.txt")
    parser.add_argument("--model_dir", type=str, default="./setting/last.pth")
    parser.add_argument("--nms_iou", type=float, default=0.3)
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--type", type=str, default="image", choices=["video", "image", "camera"])
    parser.add_argument("--show", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default="./img")
    parser.add_argument("--scale", type=float, default=1.0)
    opt = parser.parse_args()
    detector = Detector(opt)

    fs = os.listdir("./imgs")
    for f in fs:
        path = os.path.join("./imgs", f)
        detector.detect_image(path)