import os
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torchvision.transforms.functional as tf

path = "./img/train"

class Datasets(Dataset):

    def __init__(self, path, label, size, mask):
        if label == 0:
            txt_path = os.path.join(path[:-5], 'train.txt')
        elif label == 1:
            txt_path = os.path.join(path[:-3], 'val.txt')
        elif label == 2:
            txt_path = os.path.join(path[:4], 'test.txt')
        self.path = path
        # 语义分割需要的图片的图片和标签
        with open(txt_path, "r") as f:
            names = []
            for line in f.readlines():
                line = line.strip('\n')
                names.append(line)
        self.names = names
        self.size = size
        self.label = mask

    def __len__(self):
        return len(self.names)

    # 简单的正方形转换，把图片和标签转为正方形
    # 图片会置于中央，两边会填充为黑色，不会失真
    def __trans__(self, img, size, mask):
        # 图片的宽高
        h, w = img.shape[0:2]
        # 需要的尺寸
        _w = _h = size
        # 不改变图像的宽高比例
        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)
        # 缩放图像
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        # 上下左右分别要扩展的像素数
        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left
        # 生成一个新的填充过的图像，这里用纯黑色进行填充(0,0,0)
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def __getitem__(self, index):
        # 拿到的图片和标签
        name = self.names[index]
        # 图片和标签的路径
        img_path = [os.path.join(self.path, i) for i in ('DIC', self.label)]
        # 读取原始图片和标签，并转RGB
        img_o = cv2.imread(os.path.join(img_path[0], name))
        img_l = cv2.imread(os.path.join(img_path[1], name))
        img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)

        # 转成网络需要的正方形
        img_o = self.__trans__(img_o, self.size, mask=False)#256
        img_l = self.__trans__(img_l, self.size, mask=True) #256

        img_o = tf.to_tensor(img_o)
        img_l = tf.to_tensor(img_l)
        return img_o, img_l
if __name__ == '__main__':
    i = 1
    dataset = Datasets(path, 0, size=256, mask='actin')
    for a, b in dataset:
        print(i)
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        print(a)
        print(b)
        print(a.shape)
        print(b.shape)
        plt.imshow(a.transpose(1, 2, 0))
        plt.show()
        plt.imshow(b.transpose(1, 2, 0))
        plt.show()
        # save_image(a, f"./img/{i}.jpg", nrow=1)
        # save_image(b, f"./img/{i}.png", nrow=1)
        i += 1
        if i > 2:
            break
