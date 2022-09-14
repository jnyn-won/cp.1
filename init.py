## 라이브러리 불러오기
import os
import json
import numpy as np
from PIL import Image

# 경로 지정
img_path = [
#    './image/damage', 
    './image/damage_part'
    ]
lab_path = [
#    './labeling/damage',
    './labeling/damage_part'
    ]
file_names = [[name[:-5] for name in os.listdir(lab_path[N])] for N in range(len(lab_path))]
dir_data = './datasets' 

## train/test/val 폴더 생성
N = len(file_names[0])
N_train = int(N*(0.7))
N_val = int(N*0.15)
N_test = N - N_train - N_val

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

# 변환 함수들 만들기

## 두 점 사이 좌표 변환 함수
def connect(p1,p2):

    a = p1.copy()
    b = p2.copy()
    result = []
    if a[1] == b[1]:
        A,B = a[0],b[0]
        if B<A:
            A,B = B,A
        result = [[C_0, a[1]] for C_0 in range(A,B+1)]
        return result
    elif a[0] == b[0]:
        A,B = a[1],b[1]
        if B<A:
            A,B = B,A
        result = [[a[0], C_1] for C_1 in range(A,B+1)]
        return result
    grad = (b[1]-a[1])/(b[0]-a[0])
    if abs(grad) <= 1:
        if a[0] > b[0]:
            a,b = b,a
        for N in range(a[0],b[0]+1):
            Flt = grad*(N - a[0]) + a[1]
            Int = int(Flt)
            result.append([N,Int])
    elif abs(grad) > 1:
        if a[1] > b[1]:
            a,b = b,a
        for N in range(a[1],b[1]+1):
            Flt = (1/grad)*(N - a[1]) + a[0]
            Int = int(Flt)
            result.append([Int,N])
    return result

# 흑백 전환 함수
def bw(ndy):
    a = len(ndy)
    b = len(ndy[0])
    result = np.array([[0 for temp in range(b)] for temp in range(a)])
    for n1 in range(a):
        for n2 in range(b):
            result[n1][n2] = (ndy[n1][n2].sum()/3)
    return result

# 경계 내부 채우기 함수(0,255)
def filled(ndy):
    result = ndy.copy()
    for N1 in range(len(result)):
        idx_255 = np.where(result[N1] == 255)[0]
        if len(idx_255) != 0:
            if len(idx_255)%2 == 0:
                for N2 in range(len(idx_255)):
                    result[N1,idx_255[N2-1]:idx_255[N2]] = 255
    return result

## train set
for N1 in range(len(lab_path)):
    for file_name in file_names[N1][:N_train]:
        img = Image.open(os.path.join(img_path[N1], file_name + '.jpg'))
        lab = json.load(open(os.path.join(lab_path[N1], file_name + '.json')))

        c0, c1 = img.size
        label_ = np.array([[0 for temp in range(512)] for temp in range(512)])
        anns = lab['annotations']

        for ann in anns:
            if ann['part'] != None:
                seg = ann['segmentation'][0][0]
                for N2 in range(len(seg)):

                    coords = connect(
                        [int(seg[N2-1][0]*(512/c0)),
                        int(seg[N2-1][1]*(512/c1))],

                        [int(seg[N2][0]*(512/c0)),
                         int(seg[N2][1]*(512/c1))]
                        )

                    for coord in coords:
                        label_[coord[0]-1,coord[1]-1] = 255

        label_ = filled(label_)


        np.save(os.path.join(dir_save_train, f'label_{file_name}.npy'), label_)

        dummy = np.asarray(img.resize((512, 512)))
        input_ = bw(dummy)
        np.save(os.path.join(dir_save_train, f'input_{file_name}.npy'), input_)

## val set
for N1 in range(len(img_path)):
    for file_name in file_names[N1][N_train:N_train+N_val]:
        img = Image.open(os.path.join(img_path[N1], file_name + '.jpg'))
        lab = json.load(open(os.path.join(lab_path[N1], file_name + '.json')))

        c0, c1 = img.size
        label_ = np.array([[0 for temp in range(512)] for temp in range(512)])
        anns = lab['annotations']

        
        for ann in anns:
            if ann['part'] != None:
                seg = ann['segmentation'][0][0]
                for N2 in range(len(seg)):

                    coords = connect(
                        [int(seg[N2-1][0]*(512/c0)),
                        int(seg[N2-1][1]*(512/c1))],

                        [int(seg[N2][0]*(512/c0)),
                         int(seg[N2][1]*(512/c1))]
                        )

                    for coord in coords:
                        label_[coord[0]-1,coord[1]-1] = 255

        label_ = filled(label_)

        np.save(os.path.join(dir_save_val, f'label_{file_name}.npy'), label_)

        dummy = np.asarray(img.resize((512, 512)))
        input_ = bw(dummy)
        np.save(os.path.join(dir_save_val, f'input_{file_name}.npy'), input_)

## test set
for N1 in range(len(img_path)):
    for file_name in file_names[N1][N_train+N_val:]:
        img = Image.open(os.path.join(img_path[N1], file_name + '.jpg'))
        lab = json.load(open(os.path.join(lab_path[N1], file_name + '.json')))

        c0, c1 = img.size
        label_ = np.array([[0 for temp in range(512)] for temp in range(512)])
        anns = lab['annotations']

        
        for ann in anns:
            if ann['part'] != None:
                seg = ann['segmentation'][0][0]
                for N2 in range(len(seg)):

                    coords = connect(
                        [int(seg[N2-1][0]*(512/c0)),
                        int(seg[N2-1][1]*(512/c1))],

                        [int(seg[N2][0]*(512/c0)),
                         int(seg[N2][1]*(512/c1))]
                        )

                    for coord in coords:
                        label_[coord[0]-1,coord[1]-1] = 255

        label_ = filled(label_)

        np.save(os.path.join(dir_save_test, f'label_{file_name}.npy'), label_)

        dummy = np.asarray(img.resize((512, 512)))
        input_ = bw(dummy)
        np.save(os.path.join(dir_save_test, f'input_{file_name}.npy'), input_)


# ## 이미지 시각화
# import matplotlib.pyplot as plt
# plt.subplot(122)
# plt.imshow(label_, cmap='gray')
# plt.title('label')

# plt.subplot(121)
# plt.imshow(input_, cmap='gray')
# plt.title('input')

# plt.show()


# # 한 이미지의 분포
# import matplotlib.pyplot as plt
# plt.subplot(122)
# plt.hist(label_.flatten(), bins=20)
# plt.title('label')

# plt.subplot(121)
# plt.hist(input_.flatten(), bins=20)
# plt.title('input')

# plt.tight_layout()
# plt.show()

# breakpoint()
