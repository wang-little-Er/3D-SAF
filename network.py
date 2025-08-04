import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
import torch_dct as dct
from einops import rearrange
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ViT(nn.Module):  # 初始化ViT，并冻结所有权重
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained('/config')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):  # x是预处理后的图像张量，进行块嵌入和位置编码，然后提取特征
        outputs = self.model(x).last_hidden_state  # 返回[B, 197, 768]
        return outputs


class Encoder(nn.Module):  # 只使用ViT的编码器部分进行进一步的特征提取
    def __init__(self):
        super().__init__()
        self.ViT = ViT()
        self.encoder = self.ViT.model.encoder  # 从编码器开始输入,不能从ViT开始计算，因为有嵌入层
        self.layer_norm = self.ViT.model.layernorm  # 进行归一化，防止梯度爆炸

    def forward(self, features):  # features是提取的特征，保持其形状为[B,197,768]
        # print('编码器输入尺寸', features.shape)
        outputs = self.encoder(features).last_hidden_state  # 单独提取出该属性，之前没有
        outputs = self.layer_norm(outputs)
        # print('encoder', outputs.shape)
        return outputs  # # 输出的是正常ViTModel处理后的last_hidden特征


class ClsToken(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = ViT.model.pooler

    def forward(self, features):  # 此时features的形状为[B，197，768]
        cls_tokens = self.pool(features)  # 此时得到CLS方便后续进行交换,池化层会提取出CLS并进行操作
        return cls_tokens


def apply_dct_transform(image):
    """
    对RGB图像的每个通道进行DCT变换，将其转换为三通道的频域图像。
    """
    # 确保输入的形状为 [N, 3, H, W]
    N, C, H, W = image.shape
    channels = []

    for c in range(C):  # 处理 R、G、B 三个通道
        # 对每个通道进行 DCT 变换
        dct_channel = dct.dct_2d(image[:, c, :, :])  # 可能的输出形状为 [N, H, W]

        # 假设 dct_channel 是 [N, H, W]，我们需要将其调整为 [N, 1, H, W]
        dct_channel = dct_channel.unsqueeze(1)  # 增加一个通道维度，变为 [N, 1, H, W]
        channels.append(dct_channel)

    # 将处理后的通道重新组合成三通道
    dct_image = torch.cat(channels, dim=1)  # 合并为 [N, 3, H, W]
    return dct_image


class SimplifiedLinearAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        # self.dwc = nn.Sequential(nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=3,
        #                                    groups=head_dim, padding=1),
        #                          nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=3,
        #                                    groups=head_dim, padding=1)
        #                          )
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))

        print('Linear Attention window{} f{} kernel{}'.
              format(window_size, focusing_factor, kernel_size))

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        k = k + self.positional_encoding

        kernel_function = nn.ReLU()
        q = kernel_function(q)
        k = kernel_function(k)

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        with torch.cuda.amp.autocast(enabled=False):
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

            z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
            if i * j * (c + d) > c * d * (i + j):
                kv = torch.einsum("b j c, b j d -> b c d", k, v)
                x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
            else:
                qk = torch.einsum("b i c, b j c -> b i j", q, k)
                x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map

        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print('线性注意力后的形状', x.shape)
        return x

    def eval(self):
        super().eval()
        print('eval')

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                               3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        # print('注意力', x.shape)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print('交叉注意力返回X', x)
        return x


class DeepFakeDetectionModel(nn.Module):
    def __init__(self, patch_size=(16, 16), num_classes=2, norm_layer=nn.LayerNorm, num_blocks=4):  # 2/4,格外块数量
        super().__init__()

        self.num_branches = len(patch_size)  # 分支数
        self.num_classes = num_classes
        self.encoders = nn.ModuleList()
        self.encoder = Encoder()
        self.ViT = ViT()
        self.num_blocks = num_blocks

        for i in range(self.num_branches):
            self.encoders.append(self.encoder)

        self.slab1 = SimplifiedLinearAttention(768, window_size=[14, 14], num_heads=12,
                                               qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                                               focusing_factor=3, kernel_size=5)
        self.slab2 = SimplifiedLinearAttention(768, window_size=[14, 14], num_heads=12,
                                               qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                                               focusing_factor=3, kernel_size=5)

        self.cross_attention1 = CrossAttentionBlock(dim=768, num_heads=6, mlp_ratio=1., qkv_bias=False, qk_scale=None,
                                                    drop=0., attn_drop=0.,
                                                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                                    has_mlp=True)  # 为了方便，维度全部固定为768，第一版无MLP
        self.cross_attention2 = CrossAttentionBlock(dim=768, num_heads=12, mlp_ratio=1., qkv_bias=False, qk_scale=None,
                                                    drop=0., attn_drop=0.,
                                                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                                    has_mlp=True)  # 为了方便，维度全部固定为768
        self.attention_sequential = nn.ModuleList([
            CrossAttentionBlock(dim=768, num_heads=12, mlp_ratio=1., qkv_bias=False, qk_scale=None,
                                drop=0., attn_drop=0.,
                                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                has_mlp=True) for _ in range(self.num_blocks)])

        self.norm = nn.ModuleList([norm_layer(768) for _ in range(self.num_branches)])
        self.head = nn.ModuleList([nn.Linear(768, num_classes) if num_classes > 0 else nn.Identity() for _ in
                                   range(self.num_branches)])  # 分类头

    def forward_features(self, x):  # x是单张图片，分成空域和频域双分支
        """第一次特征提取"""
        xs = []  # 用于储存两个经过特征提取分支,手动储存
        tmp_img = self.ViT(x)  # 空域分支
        xs.append(tmp_img)
        # print('DCT形状', apply_dct_transform(x).shape)
        tmp_dct = self.ViT(apply_dct_transform(x))  # 频域分支
        xs.append(tmp_dct)  # 储存两分支张量
        # print('xs[0]', xs[0].shape)

        '''线性注意力机制'''
        cls_token = [x[:, 0:1] for x in xs]
        short_cut1, short_cut2 = xs[0][:, 1:, ...], xs[1][:, 1:, ...]
        xs[0] = torch.cat((cls_token[0], short_cut1 + self.slab1(xs[0][:, 1:, ...])), dim=1)
        xs[1] = torch.cat((cls_token[1], short_cut2 + self.slab1(xs[1][:, 1:, ...])), dim=1)
        # print('xs[0]', xs[0].shape)
        ##################

        """第一次特征融合"""
        cls_token.clear()
        cls_token = [x[:, 0:1] for x in xs]  # 储存不同分支的CLS
        outs = []
        for i in range(self.num_branches):  # 第一次特征融合，经过两次循环，特征分支储存在outs中
            tmp = torch.cat((cls_token[i], xs[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            # print('融合前的尺寸', tmp.shape)
            tmp = self.cross_attention1(tmp)
            outs.append(tmp)
        ####################

        """第二次特征提取"""
        xs.clear()  # 初步提取的特征已经没有了，所以列表重置为空
        for i in range(self.num_branches):
            tmp = self.encoders[i](outs[i])
            # print('tmp', tmp.shape)
            xs.append(tmp)
        ####################

        '''第二次特征融合'''
        cls_token.clear()
        cls_token = [x[:, 0:1] for x in xs]
        outs.clear()
        for i in range(self.num_branches):
            tmp = torch.cat((cls_token[i], xs[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.cross_attention2(tmp)
            outs.append(tmp)
        ####################

        '''以下全操作都只使用交叉注意力而不再进行重复进行特征提取操作'''
        for i in range(self.num_blocks):  # 交叉注意力的重复数量
            xs.clear()  # 清空临时token储存列表
            for j in range(self.num_branches):
                tmp = outs[j]
                # print('tmp', tmp.shape)
                xs.append(tmp)  # 填充token储存列表
        ###############################################
            cls_token.clear()
            cls_token = [x[:, 0:1] for x in xs]
            outs.clear()
            for j in range(self.num_branches):
                tmp = torch.cat((cls_token[j], xs[(j + 1) % self.num_branches][:, 1:, ...]), dim=1)
                tmp = self.attention_sequential[j](tmp)
                outs.append(tmp)
        '''交叉注意力序列'''

        outs = [self.norm[i](out) for i, out in enumerate(outs)]
        outs = [out[:, 0] for out in outs]  # 分开储存两个分支token的CLS 注：未加入线性注意力之前都是好用的

        return outs

    def forward(self, x):
        outs = self.forward_features(x)
        ce_logits = [self.head[i](out) for i, out in enumerate(outs)]
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)
        # print('\nce_logits:cls堆叠的形状\n', ce_logits.shape)
        return ce_logits  # 输出的是平均分类结果即多个预测CLS的均值，一张图：【0.4，0.6】


# # 数据预处理
# image_processor = ViTImageProcessor.from_pretrained('')
#
# # 选择设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = DeepFakeDetectionModel()
# model.to(device)
#
# test_image_path = ''
# image = Image.open(test_image_path).convert("RGB")
# image = image_processor(image, return_tensors="pt").to(device)
# # print(image)
# print(image["pixel_values"].shape)
#
#
# output = model(image["pixel_values"])
# print('\n网络输出\n', output)
# # print(model)
def main():
    model = DeepFakeDetectionModel(num_blocks=4)
    model.load_state_dict(torch.load(''), strict=False)
    # 定义数据集路径
    train_dir = '' 
    val_dir = ''

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # # 加载预训练模型
    # model = ViTForImageClassification.from_pretrained()

    # print('\n模型结构:\n', model)
    # 冻结特征提取部分的权重
    # for param in model.vit.parameters():
    #     param.requires_grad = False

    # 只训练分类头
    # model.classifier = nn.Linear(model.classifier.in_features, 2)  # 2类

    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    num_epochs = 20
    all_y_true = []  # 记录一个轮次中的所有标签
    all_y_scores = []  # 记录一个轮次的所有预测结果

    for epoch in range(num_epochs):
        model.train()
        print(epoch)
        total, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # print('标签', labels)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)  # 位/索引就等同于标签！
            all_y_true.extend(labels.cpu().detach().numpy())
            all_y_scores.extend(predicted.cpu().detach().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        auc_value = roc_auc_score(all_y_true, all_y_scores)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, AUC: {100 * auc_value:.2f}%， Acc: {100 * correct / total:.2f}")
        # 清空用于下一轮的列表
        all_y_true.clear()
        all_y_scores.clear()

    # 验证模型
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 索引等同于标签
            all_y_true.extend(labels.cpu().detach().numpy())
            all_y_scores.extend(predicted.cpu().detach().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 在每个epoch结束后计算AUC
        auc_value = roc_auc_score(all_y_true, all_y_scores)
    print(f"Validation Accuracy: {100 * correct / total:.2f}%, AUC: {100 * auc_value:.2f}%")

    # 保存模型权重
    torch.save(model.state_dict(), '')


def evaluate():
    all_y_true = []
    all_y_scores = []
    model = DeepFakeDetectionModel(num_blocks=4).to(device)
    model.load_state_dict(torch.load(''))
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    val_root = ''
    val_dataset = datasets.ImageFolder(val_root, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_y_true.extend(labels.cpu().detach().numpy())
            all_y_scores.extend(predicted.cpu().detach().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 在每个epoch结束后计算AUC
        auc_value = roc_auc_score(all_y_true, all_y_scores)
    print(f"Validation Accuracy: {100 * correct / total:.2f}%, AUC: {100 * auc_value:.2f}%")


if __name__ == "__main__":
    main()
    # evaluate()

