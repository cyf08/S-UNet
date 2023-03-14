import os
import numpy as np
import cv2
import pandas as pd
from skimage.measure import compare_ssim, compare_psnr

pred_path = './img/test/result/dapi'
GT_path = '../unet-pytorch-origin/img/test/dapi'
csv_path = './dapi_eval.csv'

img_names = os.listdir(pred_path)
df = pd.DataFrame(columns=['file', 'pcc', 'psnr', 'ssim'])
df.to_csv(csv_path, index=False)
print("csv file is created successfully !")

pcc = []
ssim = []
psnr = []
mse = []

for img_name in img_names:
    pred_img = cv2.imread(os.path.join(pred_path, img_name))
    GT_img = cv2.imread(os.path.join(GT_path, img_name))
    GT_img = cv2.resize(GT_img, (pred_img.shape[1], pred_img.shape[0]))
    # 计算SSIM
    ssim.append(compare_ssim(pred_img, GT_img, multichannel=True))
    print('ssim:', ssim[-1])
    # 计算PSNR
    psnr.append(compare_psnr(pred_img, GT_img))
    print('psnr:', psnr[-1])
    # 计算MSE
    # mse.append(compare_mse(pred_img, GT_img))
    # print('mse:', mse[-1])
    # 计算pcc
    pred_img = pred_img.reshape(pred_img.size, order='C')
    GT_img = GT_img.reshape(GT_img.size, order='C')
    pcc.append(np.corrcoef(pred_img, GT_img)[0, 1])
    print('pcc:', pcc[-1])
    list = [img_name, pcc[-1], psnr[-1], ssim[-1]]
    data = pd.DataFrame([list])
    data.to_csv(csv_path, mode='a', header=False, index=False)

print('mean_pcc:', np.mean(pcc))
print('mean_ssim:', np.mean(ssim))
print('mean_psnr:', np.mean(psnr))


