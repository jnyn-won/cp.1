import numpy as np
import torch

# 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

# # 트랜스폼 잘 구현되었는지 확인(시각화)
# import os
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from loader import Dataset

# dir_data = './datasets' 
# dir_save_train = os.path.join(dir_data, 'train')
# files_count = int(len(os.listdir(dir_save_train))/2)

# transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
# dataset_train = Dataset(data_dir=dir_save_train, transform=transform)
# data = dataset_train.__getitem__(np.random.randint(files_count)) # 한 이미지 불러오기
# input = data['input']
# label = data['label']

# plt.subplot(122)
# plt.hist(label.flatten(), bins=20)
# plt.title('label')

# plt.subplot(121)
# plt.hist(input.flatten(), bins=20)
# plt.title('input')

# plt.tight_layout()
# plt.show()
