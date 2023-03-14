"""
训练器模块
"""
import os
from net.SUNet import SUNet_model

import torch
import dataset
import torch.nn as nn
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
from torchvision.utils import save_image
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader

import pandas as pd
from datetime import datetime

test_path = "./img/test"
txt_path = "./img/test.txt"
mask = 'actin'
csv_name = './log/test-' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
img_size = 1024
start_value = 1
stop_value = 200
in_chans = 3
out_chans = 3

torch.backends.cudnn.benchmark = True

with open(txt_path, "r") as f:
            names = []
            for line in f.readlines():
                line = line.strip('\n')
                names.append(line)

# 训练器
class Predict:

    def __init__(self, path, model, img_save_path):
        self.path = path
        self.model = model
        self.img_save_path = img_save_path
        # 使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 网络
        self.net = SUNet_model(img_size=img_size,
                               patch_size=4,
                               in_chans=in_chans,
                               out_chans=out_chans,
                               embed_dim=96,
                               depths=[8, 8, 8, 8],
                               num_heads=[8, 8, 8, 8],
                               window_size=8,
                               mlp_ratio=4.0,
                               qkv_bias=True,
                               qk_scale=8,
                               drop_rate=0.,
                               drop_path_rate=0.1,
                               ape=False,
                               patch_norm=True,
                               use_checkpoint=False)
        self.net = self.net.to(self.device)
        self.loss_func = nn.MSELoss()
        self.loader_test = DataLoader(dataset.Datasets(test_path, 2, size=img_size, mask=mask), batch_size=1, shuffle=False, num_workers=1)
        ## GPU
        gpus = ','.join([str(i) for i in [0, 1, 2, 3]])
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        device_ids = [i for i in range(torch.cuda.device_count())]
        if torch.cuda.device_count() > 1:
            print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
        if len(device_ids) > 1:
            self.net = nn.DataParallel(self.net, device_ids=device_ids)
        # 判断是否存在模型
        if os.path.exists(self.model):
            weight = torch.load(self.model, map_location=torch.device('cpu'))
            self.net.load_state_dict(weight["state_dict"])
            try:
                self.net.load_state_dict(weight["state_dict"])
            except:
                state_dict = weight["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = 'module.' + k  # remove `module.`
                    new_state_dict[name] = v
                self.net.load_state_dict(new_state_dict)
            print(f"Loaded{self.model}!")
        else:
            print("No Param!")
            exit(1)
        os.makedirs(img_save_path, exist_ok=True)
        os.makedirs('./result/', exist_ok=True)
        df = pd.DataFrame(columns=['img', 'test Loss', 'PCC'])
        df.to_csv(csv_name, index=False)
    # 训练
    def predict(self):
        iter = 0
        # 测试
        self.net.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(self.loader_val, ascii=True, total=len(self.loader_val)):
                # 图片和分割标签
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # 输出生成的图像
                out = self.net(inputs)
                test_loss = self.loss_func(out, labels)
                img0 = out.detach().cpu().numpy()
                img1 = labels.detach().cpu().numpy()
                img0 = img0.reshape(img0.size, order='C')
                img1 = img1.reshape(img1.size, order='C')
                test_pcc = np.corrcoef(img0, img1)[0,1]
                # 保存图片
                x = inputs[0]
                x_ = out[0]
                y = labels[0]
                img = torch.cat([x, x_, y], 2)
                save_image(img.cpu(), os.path.join(self.img_save_path, f"{names[iter]}.tif"))

                test_Loss = "{:.8f}".format(test_loss)
                test_PCC = "{:.8f}".format(test_pcc)
                print(f"\npredicting {names[iter]}.tif ···, Loss: {test_Loss}, PCC: {test_PCC}")
                list = [names[iter], test_Loss, test_PCC]
                data = pd.DataFrame([list])
                data.to_csv(csv_name, mode='a', header=False, index=False)

if __name__ == '__main__':
	# 路径改一下
    t = Predict(test_path, model='./model/model_100_0.004782380238230828.plt', img_save_path=r'./img/temp')
    t.train(stop_value)
