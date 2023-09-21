import torch
import argparse
from utils.utils import get_lines, get_classes
from dataset import FRCNNDataset
from torch.utils.data import DataLoader
from trainer import FRCNNTrainer
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def train(opt):
    train_lines, valid_lines = get_lines("2007_train.txt", "2007_val.txt")
    print(f"TrainSet: {len(train_lines)}\nValidSet: {len(valid_lines)}")
    train_ds = FRCNNDataset(train_lines, opt.input_shape, show=False)
    # valid_ds = FRCNNDataset(valid_lines, opt.input_shape, show=False, train=False)
    print(f"Training on device: {torch.cuda.get_device_name(opt.id)}")
    device = "cpu" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=opt.bs,
                              drop_last=True, num_workers=0,
                              collate_fn=train_ds.collate)

    class_names, num_classes = get_classes(opt.classes_dir)
    print(f"Dataset has {num_classes} classes")

    frcnn_trainer = FRCNNTrainer(train_loader, device, opt, num_classes, class_names)

    init_epoch = 0
    if opt.pretrain:
        init_epoch = frcnn_trainer.load(opt.pretrain_model) + 1
        print("pretrain model has been loaded")

    for epoch in range(init_epoch, opt.epochs):
        frcnn_trainer.train_step(epoch)

        if (epoch + 1) % opt.test_gap == 0:
            frcnn_trainer.generate_txt(valid_lines, epoch)

            mAP = frcnn_trainer.calc_map()
            frcnn_trainer.map.append(mAP)
            frcnn_trainer.epoch.append(epoch + 1)

            if mAP > frcnn_trainer.best_map:
                frcnn_trainer.best_map = mAP
                frcnn_trainer.save(epoch, best=True)
                print("best model has been saved")

        frcnn_trainer.save(epoch, False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Faster-RCNN-Pytorch")
    parser.add_argument("--input_shape", type=int, default=600)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--classes_dir", type=str, default="voc_classes.txt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--id", type=int, default=0)

    parser.add_argument("--pretrain_backbone", type=bool, default=True)
    parser.add_argument("--freeze_backbone", type=bool, default=True)
    parser.add_argument("--freeze_epoch", type=int, default=50)
    parser.add_argument("--nms_iou", type=float, default=0.3)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--max_boxes", type=int, default=100)
    parser.add_argument("--map_out_path", type=str, default="./map_out")
    parser.add_argument("--MINOVERLAP", type=float, default=0.5)
    parser.add_argument("--draw_plot", type=bool, default=True)
    parser.add_argument("--pretrain", type=bool, default=False)
    parser.add_argument("--pretrain_model", type=str, default="")


    parser.add_argument("--ratios", type=list, default=[0.5, 1, 2])
    parser.add_argument("--scales", type=list, default=[8, 16, 32])
    parser.add_argument("--scale", type=float, default=1.0)

    parser.add_argument("--save_dir", type=str, default="./setting")
    parser.add_argument("--test_gap", type=int, default=10)
    opt = parser.parse_args()
    train(opt)