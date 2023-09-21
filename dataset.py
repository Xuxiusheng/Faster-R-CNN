from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import torch


class FRCNNDataset(Dataset):
    def __init__(self, lines, input_shape=600, train=True, show=False):
        self.lines = lines
        self.input_shape = input_shape
        self.train = train
        self.prob = 0.5
        self.show = show

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = self.lines[item]
        data = line.split(" ")
        image = Image.open(data[0])
        w, h = image.size
        nh, nw = self.input_shape, self.input_shape
        boxes = np.array([np.array(list(map(int, box.split(',')))) for box in data[1:]])
        if self.train:
            c = random.random()
            if c < self.prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        ratio = min(nw / w, nh / h)
        iw, ih = int(w * ratio), int(h * ratio)
        dx = (nw - iw) // 2
        dy = (nh - ih) // 2
        image = image.resize((iw, ih), Image.BICUBIC)
        new_image = Image.new("RGB", (nw, nh), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = np.array(new_image, dtype=np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * iw / w + dx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * ih / h + dy

        boxes[:, :2][boxes[:, :2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > nw] = nw
        boxes[:, 3][boxes[:, 3] > nh] = nh

        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]

        if self.show:
            font = ImageFont.truetype(font='simhei.ttf',
                                      size=np.floor(3e-2 * image.shape[1] + 0.5).astype('int32'))
            thickness = int(max((image.shape[0] + image.shape[1]) // np.mean(self.input_shape), 1))

            for i, box in enumerate(boxes):
                xmin, ymin, xmax, ymax, c = box
                text = "object"
                draw = ImageDraw.Draw(new_image)
                text_size = draw.textsize(text, font)
                text = text.encode("utf-8")
                text_origin = np.array([xmin, ymin + 1])

                for i in range(thickness):
                    draw.rectangle([xmin + i, ymin + i, xmax - i, ymax - i])
                draw.rectangle([tuple(text_origin), tuple(text_origin + text_size)], outline=(255, 0, 0))
                draw.text(text_origin, str(text, 'UTF-8'), fill=(0, 0, 0), font=font)
            new_image.show()

        image /= 255.0
        image = np.transpose(image, (2, 0, 1))

        box_data = np.zeros((len(boxes), 5))
        if len(boxes) > 0:
            box_data[:len(boxes)] = boxes
        boxes = box_data[:, :4]
        label = box_data[:, -1]

        return image, boxes, label

    def collate(self, batch):
        images = []
        bboxes = []
        labels = []

        for img, box, label in batch:
            images.append(img)
            bboxes.append(box)
            labels.append(label)
        images = torch.from_numpy(np.array(images))
        return images, bboxes, labels
