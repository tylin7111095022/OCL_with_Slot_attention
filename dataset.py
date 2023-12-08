import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ChromosomeDataset(Dataset):
    f'''
    回傳img(torch.uint8)及label(torch.float32)\n
    這個dataset 最後回傳的圖片為RGB圖
    '''
    def __init__(self,img_dir,mask_dir,imgsize:int=0,transform = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgsize = imgsize
        self.img_list = [i for i in os.listdir(img_dir)]
        self.transforms = transform

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_a = _load_img(os.path.join(self.img_dir, img_name),size=self.imgsize,interpolation=cv2.INTER_CUBIC,rgb=True)
        if img_a.ndim == 2:
            img_a = np.expand_dims(img_a,2) #如果沒通道軸，加入通道軸
        img = torch.permute(torch.from_numpy(img_a),(2,0,1))
        img = ((img / 255.0) - 0.5) * 2.0  # Rescale to [-1, 1].
        img = torch.clamp(img,-1.0,1.0)
        imgsize = (img.shape[1], img.shape[2])

        if self.mask_dir:
            label_name = img_name
            label_path = os.path.join(self.mask_dir, label_name)
            label = _load_img(label_path,size=self.imgsize,interpolation=cv2.INTER_CUBIC,rgb=False)

            if label.ndim == 2:
                label = np.expand_dims(label,2) #如果沒通道軸，加入通道軸 為了執行後面的cv2.threshold
            label = cv2.resize(label, (imgsize[1],imgsize[0]), cv2.INTER_NEAREST) #將label resize成跟 img  一樣的長寬
            #label內的值不只兩個，這導致除以255後值介於0~1的值在後續計算iou將label轉回int64的時候某些值被無條件捨去成0
            ret,label_binary = cv2.threshold(label,127,255,cv2.THRESH_BINARY)
            label_t = torch.from_numpy(label_binary).unsqueeze(0).to(torch.float32)#加入通道軸
            # 處理標籤，将像素值255改為1
            if label_t.max() > 1:
                label_t[label_t == 255] = 1     
        else:
            label_t = torch.zeros_like(img, dtype = torch.float32)

        if self.transforms:
            img = self.transform(img)
            label_t = self.transform(label_t)
            

        return img, label_t

    def __len__(self):
        return len(self.img_list)
    
class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

    
def get_num(names):
    names = names.split(".")[0]
    names = names.split("_")
    for i,n in enumerate(names):
        if i == 0:
            min_len = len(n)
            min_len_ndx = i
        else:
            if len(n) <= min_len:
                min_len = len(n)
                min_len_ndx = i

    return int(names[min_len_ndx])

def _load_img(file, size, interpolation, rgb:bool):
    """size可以為正整數或None或0"""
    if rgb:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if size: #如果size為正整數
        img = cv2.resize(img, (size,size), interpolation)
    return np.asarray(img, np.float32)

if __name__ == "__main__":
    ds = ChromosomeDataset(img_dir = r'dataset\zong\20000\images', mask_dir= r'dataset\zong\20000\masks\255',imgsize=150)
    # for i in range(len(ds)):
    #     img, mask = ds[i]
    #     print(img.shape)
    #     print(mask.shape)

    img, mask= ds[500]
    print(img.shape)
    print(mask.shape)

    cv2.imwrite("img.jpg", img.permute(1,2,0).numpy())
    cv2.imwrite("mask.jpg", (mask*255).permute(1,2,0).numpy(),)