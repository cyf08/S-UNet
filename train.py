import os
from net.SUNet import SUNet_model

import torch
import dataset
import torch.nn as nn
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import utils.draw_loss as draw
from torchvision.utils import save_image
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader

import pandas as pd
from datetime import datetime
from torchsummary import summary

path = "./img/train"
val_path = "./img/val"
base_lr = 2e-4
min_lr = 1e-6
lr_set = base_lr
batch_size = 2
num_workers = 0
mask = 'actin'
csv_name = './log/Loss-' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'
pretrained_model = './model_bestPSNR.pth'
img_size = 256
start_value = 1
stop_value = 200
in_chans = 3
out_chans = 3

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


# 训练器
class Trainer:

    def __init__(self, path, val_path, model, model_copy, img_save_path):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path
        self.val_path = val_path
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
        summary(self.net,(3, 256, 256),batch_size=1,device="cpu")
        self.net = self.net.to(self.device)
        # 优化器，这里用的Adam，跑得快点
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr_set, betas=(0.9, 0.999), eps=1e-8)
        # 这里直接使用二分类交叉熵来训练
        self.loss_func = nn.L1Loss()
        # 设备好，batch_size和num_workers可以给大点

        self.loader = DataLoader(dataset.Datasets(path, 0, size=img_size, mask=mask), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.loader_val = DataLoader(dataset.Datasets(val_path, 1, size=img_size, mask=mask), batch_size=1, shuffle=False, num_workers=num_workers)
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
        if os.path.exists(pretrained_model):
            if self.device == 'cuda':
                weight = torch.load(pretrained_model)
            else:
                weight = torch.load(pretrained_model, map_location=torch.device('cpu'))
            # self.net.load_state_dict(weight["state_dict"])
            try:
                self.net.load_state_dict(weight["state_dict"])
            except:
                state_dict = weight["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = 'module.' + k  # remove `module.`
                    new_state_dict[name] = v
                self.net.load_state_dict(new_state_dict)
            print(f"Loaded{pretrained_model}!")
        else:
            print("No Param!")
        os.makedirs(img_save_path, exist_ok=True)
        os.makedirs('./log', exist_ok=True)
        df = pd.DataFrame(columns=['time', 'step', 'train Loss', 'val Loss', 'PCC', 'val PCC'])
        df.to_csv(csv_name, index=False)
    # 训练
    def train(self, stop_value):

        epoch = start_value
        max_iterations = stop_value * len(self.loader)
        while epoch <= stop_value:
            torch.cuda.empty_cache()
            total_loss = 0
            total_valloss = 0
            pccs = []
            valpccs = []
            self.net.train()
            for inputs, labels in tqdm(self.loader, desc=f"Epoch {epoch}/{stop_value}",
                                       ascii=True, total=len(self.loader)):
                # 图片和分割标签
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # 输出生成的图像
                inputs = inputs
                out = self.net(inputs)
                loss = self.loss_func(out, labels)
                # 后向
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                lr_set = base_lr * (1.0 - (epoch-1) / max_iterations) ** 0.9
                for param_group in self.opt.param_groups:
                    param_group['lr'] = lr_set
                total_loss += loss.item()

                img0 = out.detach().cpu().numpy()
                img1 = labels.detach().cpu().numpy()
                img0 = img0.reshape(img0.size, order='C')
                img1 = img1.reshape(img1.size, order='C')
                pcc = np.corrcoef(img0, img1)[0,1]
                pccs.append(pcc)

                if epoch % 10 == 0:
                    x = inputs[0]
                    x_ = out[0]
                    y = labels[0]
                    img = torch.cat([x, x_, y], 2)
                    save_image(img.cpu(), os.path.join(self.img_save_path, f"{epoch}.tif"))
                    # print("image save successfully !")
            loss = total_loss / len(self.loader)
            pcc_r = np.mean(pccs).astype(np.float32)
            torch.save(self.net.state_dict(), self.model)
            # print("model is saved !")

            # 验证
            self.net.eval()
            with torch.no_grad():
                for inputs, labels in tqdm(self.loader_val, desc=f"Epoch {epoch}/{stop_value}",
                                        ascii=True, total=len(self.loader_val)):
                    # 图片和分割标签
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # 输出生成的图像
                    out = self.net(inputs)
                    valloss = self.loss_func(out, labels)
                    total_valloss += valloss.item()

                    img0 = out.detach().cpu().numpy()
                    img1 = labels.detach().cpu().numpy()
                    img0 = img0.reshape(img0.size, order='C')
                    img1 = img1.reshape(img1.size, order='C')
                    val_pcc = np.corrcoef(img0, img1)[0,1]
                    valpccs.append(val_pcc)
            val_loss = total_valloss / len(self.loader_val)
            valpcc_r = np.mean(valpccs).astype(np.float32)
            Loss = "{:.8f}".format(loss)
            val_Loss = "{:.8f}".format(val_loss)
            PCC = "{:.8f}".format(pcc_r)
            val_PCC = "{:.8f}".format(valpcc_r)
            print(f"\nEpoch: {epoch}/{stop_value}, learning rate: {lr_set}, Loss: {Loss}, PCC: {PCC}, valLoss: {val_Loss}, valPCC: {val_PCC}")
            time = "%s"%datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            list = [time, epoch, Loss, val_Loss, PCC, val_PCC]
            data = pd.DataFrame([list])
            data.to_csv(csv_name, mode='a', header=False, index=False)
            # 绘制loss曲线
            draw.draw_loss(csv_name)  

            # 备份
            if epoch % 100 == 0:
                # torch.save(self.net.state_dict(), self.model_copy.format(epoch, loss))
                print("model_copy is saved !")

            epoch += 1


if __name__ == '__main__':
	# 路径改一下
    t = Trainer(path, val_path, r'./model/model.pth', r'./model/model_{}_{}.pth', img_save_path=r'./img/temp')
    t.train(stop_value)
