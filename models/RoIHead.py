import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool

warnings.filterwarnings("ignore")

def normal_init(m, mean, stddev):
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()


class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        self.cls_loc = nn.Linear(2048, n_class * 4)
        self.score = nn.Linear(2048, n_class)
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size, device):
        n, _, _, _ = x.shape
        roi_indices = roi_indices.to(device)
        rois = rois.to(device)

        rois = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.shape[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.shape[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)

        pool = self.roi(x, indices_and_rois)

        fc7 = self.classifier(pool)
        fc7 = fc7.view(fc7.size(0), -1)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.shape[1])
        roi_scores = roi_scores.view(n, -1, roi_scores.shape[1])
        return roi_cls_locs, roi_scores

