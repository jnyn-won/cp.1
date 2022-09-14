import os
import numpy as np

import torch

# 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 정규화
        label = label/255.0
        input = input/255.0

        # 이미지와 레이블의 차원 = 2일 경우(채널이 없을 경우, 흑백 이미지), 새로운 채널(축) 생성
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        # transform이 정의되어 있다면 transform을 거친 데이터를 불러옴
        if self.transform:
            data = self.transform(data)

        return data



# # 데이터로더 잘 구현되었는지 확인(시각화)
# import matplotlib.pyplot as plt
# dir_data = './datasets' 
# dir_save_train = os.path.join(dir_data, 'train')
# files_count = int(len(os.listdir(dir_save_train))/2)

# dataset_train = Dataset(data_dir=dir_save_train)
# data = dataset_train.__getitem__(np.random.randint(files_count))
# input = data['input']
# label = data['label']

# plt.subplot(122)
# plt.imshow(label, cmap='gray')
# plt.title('label')

# plt.subplot(121)
# plt.imshow(input, cmap='gray')
# plt.title('input')

# plt.show()
