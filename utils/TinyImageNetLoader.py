import torch
import os
import torchvision.transforms as transforms
from torchvision import datasets

def tiny_loader(batch_size, data_dir):
    num_label = 200
    # normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(64), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
          ])
        #   normalize,
    transform_test = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader



from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.label_to_index = {}
        self.transform = transform

        # 初始化标签到索引的转换字典
        
        # 初始化标签到索引的转换字典，使用排序确保一致性
        sorted_categories = sorted(os.listdir(root_dir))
        for idx, category in enumerate(sorted(os.listdir(root_dir))):
            self.label_to_index[category] = idx 

        # 遍历所有类别的目录
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            images_folder_path = os.path.join(category_path, "images")  # 添加 images 文件夹路径
            if os.path.isdir(images_folder_path):  # 检查 images 文件夹是否存在
                # 遍历类别目录中的所有图片
                for img_name in os.listdir(images_folder_path):
                    img_path = os.path.join(images_folder_path, img_name)
                    if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                        self.images.append(img_path)
                        # 保存数字标签而不是字符串标签
                        self.labels.append(self.label_to_index[category])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label  # 返回图像和数字标签



# transformations = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# dataset = CustomImageDataset(root_dir='/home/lym/code/data/tiny-imagenet-200/val/', transform=transformations)
# loader = DataLoader(dataset, batch_size=10, shuffle=False)

# # 检查是否成功加载了图片和标签
# for images, labels in loader:
#     print(images.shape, labels)