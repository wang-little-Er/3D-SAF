import torch
import torch.nn as nn
from einops import rearrange
from torchvision import datasets, transforms
import torch_dct as dct  # 用于频域变换的库
import torch.optim as optim
from 参考工作 import DeepFakeDetectionModel
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch.hub
from timm.models.vision_transformer import vit_base_patch8_224_in21k as create_model
import math
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from utils import read_split_data, train_one_epoch, evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transform = {# transforms.Normalize(mean=[0.486, 0.457, 0.408], std=[0.230, 0.224, 0.229])
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 ]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.486, 0.457, 0.408], std=[0.230, 0.224, 0.229])])}


def apply_dct_transform(image):
    """
    对RGB图像的每个通道进行DCT变换，将其转换为三通道的频域图像。
    """
    # 将RGB图像的每个通道分别提取并应用DCT变换
    channels = []
    for c in range(3):  # 处理 R、G、B 三个通道
        single_channel = image[c, :, :].unsqueeze(0)  # 提取单通道并增加维度
        dct_channel = dct.dct_2d(single_channel)  # 对单通道进行DCT变换
        channels.append(dct_channel)

    # 将处理后的通道重新组合成三通道
    dct_image = torch.cat(channels, dim=0)  # 合并为三通道 [3, H, W]
    return dct_image


class DeepFakeDataset(Dataset):
    def __init__(self, images_path, images_class, transform):  # 完整的图片路径，以及它们对应的标签列表
        self.images = images_path
        self.labels = images_class
        self.data_transforms = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        image = self.data_transforms(image)
        label = self.labels[index]
        dct_image = apply_dct_transform(image)
        return image, dct_image, label


if __name__ == '__main__':
    options = ['train', 'eval']
    option = options[0]
    # layer_num = 11  # 一共16层
    data_path = 'G:\PyTorchProgramming/202204\ch6cnn_demo\TFFF-main\exchange_faces/200张的原始测试图'
    tb_writer = SummaryWriter()
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)
    # 加载数据集
    # 实例化训练数据集
    train_dataset = DeepFakeDataset(images_path=train_images_path,
                                    images_class=train_images_label,
                                    transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = DeepFakeDataset(images_path=val_images_path,
                                  images_class=val_images_label,
                                  transform=data_transform["val"])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)  # 16
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)

    # 初始化网络模型
    model = DeepFakeDetectionModel().to(device)
    if option == 'train':
        print("\ntraining...")
        # pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5E-5)  # 传入需要进行SGD的参数组成的dict0.01
        # lr=0.001
        lrf = 0.01  # 0.01
        epochs = 200
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (  # 10轮
                1 - lrf) + lrf  # cosine learning rate decay
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # cosine learning rate decay

        for epoch in range(epochs):
            # train   返回 平均loss 和 预测正确的样本÷样本总数
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)
            # validate
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)
            scheduler.step()
            # 以上循环内三部分写法源于LambdaLR()的torch官方实例
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
    else:
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=9)
