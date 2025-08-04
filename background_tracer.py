import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from scipy.spatial.distance import cosine


class IdentityDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # 标准化步骤不可缺少
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_paths = []
        self.labels = []
        # 遍历所有子目录，并加载图像路径和标签
        for class_index, class_name in enumerate(os.listdir(image_dir)):
            class_path = os.path.join(image_dir, class_name)
            if os.path.isdir(class_path):  # 只处理目录
                for img_file in os.listdir(class_path):
                    if img_file.endswith('.jpg') or img_file.endswith('.png'):
                        self.image_paths.append(os.path.join(class_path, img_file))
                        self.labels.append(class_index)  # 使用目录索引作为标签  文件路径，目录的索引作为每张图像的标签
        self.unique_labels = list(set(self.labels))

    def __len__(self):
        print(f"Total samples: {len(self.image_paths)}")
        return len(self.image_paths)

    def __getitem__(self, idx):
        img1_path = self.image_paths[idx]
        img1_label = self.labels[idx]
        # 随机选择图像对：同类或不同类
        if random.random() > 0.5:
            # 选择相同标签的图像
            same_label_indices = [i for i, label in enumerate(self.labels) if label == img1_label and i != idx]
            if same_label_indices:
                img2_idx = random.choice(same_label_indices)
            else:
                img2_idx = idx  # 默认选择自己
            label = 1  # 相同标签
        else:
            # 选择不同标签的图像
            different_label = random.choice([l for l in self.unique_labels if l != img1_label])
            different_label_indices = [i for i, label in enumerate(self.labels) if label == different_label]
            img2_idx = random.choice(different_label_indices)
            label = -1  # 不同标签
        img2_path = self.image_paths[img2_idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label


# 身份匹配系统类
class IdentityMatchingSystem:
    def __init__(self, model_name='config', weights_path=None):
        self.image_paths = None
        self.feature_database = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)

        # 使用训练好的大数据下的初始化权重，然后冻结一部分权重，在小数据集上微调
        for param in self.model.parameters():
            param.requires_grad = True

        # 冻结embeddings层
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

        # 冻结前6层encoder blocks
        for i in range(6):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = False

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # 小学习率是好选择
        self.loss_fn = nn.CosineEmbeddingLoss()

        # 如果提供了权重路径，加载预训练权重
        if weights_path:
            self.load_weights(weights_path)

    def load_weights(self, weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")

    def extract_features(self, img):
        # 在训练过程中，我们希望计算梯度，所以不使用 torch.no_grad()
        if isinstance(img, torch.Tensor):
            # 假设输入是 [batch_size, channels, height, width] 格式的张量
            img = img.cpu().numpy().transpose(0, 2, 3, 1)  # 转换为 [batch_size, height, width, channels]
            img = (img * 255).astype(np.uint8)  # 转换回 0-255 范围
            img = [Image.fromarray(i) for i in img]  # 转换为 PIL 图像列表
        with torch.no_grad():  # 训练时去掉
            inputs = self.feature_extractor(images=img, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # 获取[CLS]特征
        return features

    def train(self, dataloader, epochs=1):
        self.model.train()  # 确保模型处于训练模式
        for epoch in range(epochs):
            total_loss = 0
            for img1, img2, label in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                img1, img2 = img1.to(self.device), img2.to(self.device)
                label = label.float().to(self.device)  # 确保标签是浮点类型

                self.optimizer.zero_grad()

                output1 = self.extract_features(img1)
                output2 = self.extract_features(img2)

                loss = self.loss_fn(output1, output2, label)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
        # 保存模型权重
        torch.save(self.model.state_dict(), 'model_weights_contrastive24-12-4.pth')

    def build_feature_database(self, root_dir):
        self.model.eval()
        self.feature_database = []
        self.image_paths = []

        print("开始构建特征数据库...")

        # 遍历根目录中的每个类别文件夹
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                # 读取当前类别文件夹中的第一张图像
                for img_file in os.listdir(class_path):
                    if img_file.endswith('.jpg') or img_file.endswith('.png'):
                        img_path = os.path.join(class_path, img_file)
                        img = Image.open(img_path).convert('RGB')
                        # 提取特征
                        feature = self.extract_features(img)
                        self.feature_database.append(feature.cpu())
                        self.image_paths.append(img_path)
                        print(f"已将特征保存到库，图像路径: {img_path}")
                        # 只处理每个类别的第一张图像
                        # break

        self.feature_database = torch.cat(self.feature_database, dim=0).to(self.device)

        print(f"特征库构建完成，特征数量: {len(self.feature_database)}")
        return self.feature_database, self.image_paths

    import numpy as np
    import matplotlib.pyplot as plt

    def visualize_features(self, query_img, query_feature, best_match_feature, similar_image_path):
        # 将特征转换为 NumPy 数组
        query_feature_np = query_feature.squeeze().cpu().numpy()
        best_match_feature_np = best_match_feature.squeeze().detach().cpu().numpy()

        # 计算余弦相似度
        cosine_similarity = 1 - cosine(query_feature_np, best_match_feature_np)

        # 计算合适的图像尺寸
        feature_length = len(query_feature_np)
        side_length = int(math.ceil(math.sqrt(feature_length)))

        # 将特征填充到方形数组中
        query_feature_img = np.zeros((side_length * side_length))
        query_feature_img[:feature_length] = query_feature_np
        query_feature_img = query_feature_img.reshape((side_length, side_length))

        best_match_feature_img = np.zeros((side_length * side_length))
        best_match_feature_img[:feature_length] = best_match_feature_np
        best_match_feature_img = best_match_feature_img.reshape((side_length, side_length))

        # 计算差异
        difference = query_feature_img - best_match_feature_img

        # 创建长图
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle('Feature Comparison', fontsize=16)

        # 1. 查询图像
        axs[0].imshow(query_img)
        axs[0].set_title('Query Image')
        axs[0].axis('off')

        # 2. 查询特征
        im1 = axs[1].imshow(query_feature_img, cmap='viridis')
        axs[1].set_title('Query Feature')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        # 3. 最佳匹配特征
        im2 = axs[2].imshow(best_match_feature_img, cmap='viridis')
        axs[2].set_title('Best Match Feature')
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        # 4. 特征差异图
        im3 = axs[3].imshow(difference, cmap='coolwarm', vmin=-np.max(np.abs(difference)),
                            vmax=np.max(np.abs(difference)))
        axs[3].set_title(f'Feature Difference\nCosine Similarity: {cosine_similarity:.4f}')
        plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

        # 5. 匹配结果图
        similar_image = plt.imread(similar_image_path)
        axs[4].imshow(similar_image)
        axs[4].set_title(f'Best Match:\n{os.path.basename(similar_image_path)}')
        axs[4].axis('off')

        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 保存图像
        plt.savefig('feature_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def find_similar_identity(self, query_img):
        self.model.eval()
        with torch.no_grad():
            query_feature = self.extract_features(query_img)

            # 确保特征数据库在同一设备上
            if self.feature_database.device != self.device:
                self.feature_database = self.feature_database.to(self.device)

            # 计算相似度
            similarities = torch.nn.functional.cosine_similarity(query_feature, self.feature_database)

        # 找到最相似的图像
        max_similarity, max_index = torch.max(similarities, dim=0)
        similar_image_path = self.image_paths[max_index]

        # 修改这里，传入 query_img
        self.visualize_features(query_img, query_feature, self.feature_database[max_index], similar_image_path)

        return similar_image_path, max_similarity.item()


# 使用示例
if __name__ == "__main__":
    # 设置数据转换
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ]) 

    # 创建数据集 
    dataset = IdentityDataset("", transform=None)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)  

    # 初始化身份匹配系统
    system = IdentityMatchingSystem(weights_path='model_weights_contrastive24-12-4.pth')

    # 读取训练好的权重

    # system.load_state_dict(torch.load('model_weights_contrastive.pth'))
    
    # 训练模型
    system.train(dataloader, epochs=200)  

    # 构建特征数据库 
    system.build_feature_database("")

    # 测试查询
    query_img_path = ""
    query_img = Image.open(query_img_path).convert('RGB')
    query_img.show()
    # query_img = transform(query_img)

    similar_image_path, similarity_score = system.find_similar_identity(query_img)
    # print(f"Most similar identity: {similar_label}")
    print(f"Similarity score: {similarity_score:.4f}")
    print(f"Similar image path: {similar_image_path}")

    # 显示最相似的图像（可选）
    # similar_image = Image.open(similar_image_path)
    # similar_image.show()


