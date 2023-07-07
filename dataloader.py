import glob
import torch
from torch.utils.data import Dataset, DataLoader
from medpy.io import load, save
from scipy.io import savemat, loadmat
from torchvision import transforms 
import numpy as np

device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")
print('device is {}'.format(device))

train_root = '/content/BRATS2015_Training_Processed/train/'
validation_root =  '/content/BRATS2015_Training_Processed/test/'

mdict = loadmat("/content/mriRecon/PosEncode.mat")
PosEncoding = mdict["PositionalEncoding"]
PosEncoding = torch.tensor(PosEncoding)

transform_2015 = transforms.Compose([transforms.ConvertImageDtype(torch.float),
                                     transforms.Normalize(0.5, 0.5, 0.5)])

def cal_fft_for_all_channels(x):
    fft = torch.zeros_like(x,dtype=torch.complex64)
    fft = torch.fft.fft2(x)
    fft = torch.fft.fftshift(fft,dim=(1,2))
    return torch.real(fft).to(torch.float32) ,torch.imag(fft).to(torch.float32)

class Brats2013_2D(Dataset):
    def __init__(self, root, transform=transform_2015 , PE = PosEncoding):
        self.img_dir = root
        self.positionalEncoding = PE
        self.transform = transform
        file_list = glob.glob(root + "*.mha")
        self.data = []
        for file_path in file_list:
            layer_num = file_path.split("\\")[-1].split(".")[-2].split("_")[-1]
            subject_num = file_path.split("/")[-1].split(".")[-2].split("_")[-2]
            self.data.append([file_path, int(layer_num), int(subject_num)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, LayerNum, subjectNum = self.data[idx]
        data, image_header = load(img_path)
        data = np.transpose(data,(2,0,1))
        image = torch.tensor(data[0:6,:,:]).to(device).to(torch.float32)
        label = torch.tensor(data[6,:,:][np.newaxis,:,:]).to(device).to(torch.float32)
        image_fft_r,image_fft_i = cal_fft_for_all_channels(image)
        image_fft_r = torch.tensor(image_fft_r)
        image_fft_i = torch.tensor(image_fft_i)
        image = torch.tensor(image)
        return image_fft_r, image_fft_i, image, label, torch.tensor(LayerNum), torch.tensor(subjectNum), torch.tensor(self.positionalEncoding)
    
    
dataset_val = Brats2013_2D(root=validation_root , PE= PosEncoding)
dataloader_val = DataLoader(dataset_val, batch_size=10, shuffle=True,pin_memory=False,drop_last=True)
