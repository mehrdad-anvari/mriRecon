import torch
from torch import nn
import dataloader as dl
import os
import models
import utils
import torch.optim as optim
import numpy as np
import torchvision.utils as vutils
from scipy.io import savemat
from medpy.io import load, save


percent = 40
kmodel_dir1 = '/content/mriRecon/trainedModels/modelt1.pt'
kmodel_dir2 = '/content/mriRecon/trainedModels/modelt2.pt'
save_to = f'/content/drive/MyDrive/results/{percent}/'

unetT1_dir = f'/content/drive/MyDrive/trainedModels/Unet/T1/{percent}/model_unet.pt'
unetT2_dir = f'/content/drive/MyDrive/trainedModels/Unet/T2/{percent}/model_unet.pt'

mask = utils.load_mask(f'GaussianDistribution1DMask_{percent}_257')
# mask = load_mask('UniformDistribution1DMask_30_257')
# mask = load_mask('Deterministic1DMask_40_257')
mask_c = 1 - mask

ngpu = 1

torch.cuda.empty_cache()

batch_size = 10
dataset_val = dl.Brats2013_2D(root=dl.validation_root , PE= dl.PosEncoding)
dataloader_val = dl.DataLoader(dataset_val, batch_size=batch_size, shuffle=True,pin_memory=False,drop_last=True)


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

image_fft_r, image_fft_i, image, LayerNum, subjectNum, PosEncoding  = next(iter(dataloader_val))

mask = torch.tensor(mask).to(device)
mask_c = torch.tensor(mask_c).to(device)





## Define the model
print('Define Models..')
netG1 = models.u_net_bn(ngpu).to(device)
netG1 = netG1.float()
netG2 = models.u_net_bn(ngpu).to(device)
netG2 = netG2.float()
model2 = models.KspaceNetT1(ngpu).to(device)
model1 = models.KspaceNetT1(ngpu).to(device)

## Load the model if exists, pre-trained model
if  os.path.isfile(kmodel_dir2):
    model2 = torch.load(kmodel_dir2)
    print('K-spaceT2 model loaded!')   
model2.eval()
    
if  os.path.isfile(kmodel_dir1):
    model1 = torch.load(kmodel_dir1)
    print('K-spaceT1 model loaded!')   
model1.eval()

if  os.path.isfile(unetT1_dir):
    netG1 = torch.load(unetT1_dir)
    print('Unet-T1 model loaded!')
netG1.eval()
    
if  os.path.isfile(unetT2_dir):
    netG2 = torch.load(unetT2_dir)
    print('Unet-T2 model loaded!')
netG2.eval()

def infer_using_KspaceNet(image_fft_r,image_fft_i,LayerNum,model,mask,mask_c,t_c,t_p,PosEncoding):
    for i in range(6):
            image_fft_r[:,i,:,:] = image_fft_r[:,i,:,:].to(device) * mask[i,:,:]
            image_fft_i[:,i,:,:] = image_fft_i[:,i,:,:].to(device) * mask[i,:,:]
    ListConChIdxs = []
    if t_c == 1:
        ListConChIdxs = [0,t_c,2,3,5] ## index 1 is the T1 middle layer that should be predicted
    if t_c == 4:
        ListConChIdxs = [0,2,3,t_c,5]
    image_fft_in = torch.concat((image_fft_r[:,ListConChIdxs,:,:],image_fft_i[:,ListConChIdxs,:,:]), dim = 1).to(device)
    image_fft_out = torch.concat((image_fft_r[:,t_p,:,:][:,np.newaxis,:,:],image_fft_i[:,t_p,:,:][:,np.newaxis,:,:]), dim = 1).to(device)
    ## Inference
    Predition = model(image_fft_in.float(),PosEncoding.float(),LayerNum.to(device))
    
    # correct values of estimated image using true ones
    Predition[:,0,:,:] = Predition[:,0,:,:] * mask_c[t_p,:,:] + image_fft_r[:,t_p,:,:].to(device) * mask[t_p,:,:]
    Predition[:,1,:,:] = Predition[:,1,:,:] * mask_c[t_p,:,:] + image_fft_i[:,t_p,:,:].to(device) * mask[t_p,:,:]
    
    complexPrediction  = Predition[:,0,:,:][:,np.newaxis,:,:]+Predition[:,1,:,:][:,np.newaxis,:,:]*1j
    complexPrediction = torch.fft.ifftshift(complexPrediction,dim=(2,3))
    image_Predition = torch.fft.ifft2(complexPrediction)
    image_Predition = torch.abs(torch.tensor(image_Predition).float())
    return image_Predition

def cal_losses(fake1,good_imgs1,Pred_slice):
    fake_fft2 = torch.fft.fft2(fake1)
    good_img_fft2 = torch.fft.fft2(good_imgs1)

    fake_fft2 = torch.fft.fftshift(fake_fft2) * mask_c[Pred_slice,:,:] + torch.fft.fftshift(good_img_fft2) * mask[Pred_slice,:,:]
    fake_fft2 = torch.fft.ifftshift(fake_fft2)
    fake1 = torch.fft.ifft2(fake_fft2).float()
    
    errG_fft_mse_r = torch.mean(torch.square(fake_fft2.real - good_img_fft2.real)) * 0.02
    errG_fft_mse_i = torch.mean(torch.square(fake_fft2.imag - good_img_fft2.imag)) * 0.02
    errG_fft = errG_fft_mse_r + errG_fft_mse_i

    errG_numerator = torch.sqrt(torch.sum(torch.square(fake1 - good_imgs1)))
    errG_denominator = torch.sqrt(torch.sum(torch.square(good_imgs1)))
    errG_mse = (errG_numerator/errG_denominator)*15
    
    return errG_fft, errG_mse, fake1

def save_slices(fakeT1,T1,fakeT2,T2,label,subjectNum,layerNum,root,percent):
    dataUnit = np.zeros((257,257,5))
    for i in range(fakeT1.shape[0]):
        dataUnit[:,:,0]=fakeT1[i,0,:,:].detach().cpu().numpy()
        dataUnit[:,:,1]=T1[i,0,:,:].detach().cpu().numpy()
        dataUnit[:,:,2]=fakeT2[i,0,:,:].detach().cpu().numpy()
        dataUnit[:,:,3]=T2[i,0,:,:].detach().cpu().numpy()
        dataUnit[:,:,4]=label[i,0,:,:].detach().cpu().numpy()
        save(dataUnit, root +  f'{subjectNum[i]}_{layerNum[i]}.mha', use_compression = True)
        
with torch.no_grad():
    model1.train(False)
    model2.train(False)
    netG1.train(False)
    netG2.tarin(False)
    for val_iter, data in enumerate(dataloader_val,0):

        image_fft_r, image_fft_i, image, LayerNum, subjectNum, PosEncoding = data
        image = image.to(device)
        image_fft_i = image_fft_i.to(device)
        image_fft_r = image_fft_r.to(device)

        T1_pred = infer_using_KspaceNet(image_fft_r,image_fft_i,LayerNum,model=model1,mask=mask,mask_c=mask_c,t_c=4,t_p=1,PosEncoding=PosEncoding.to(device))
        T2_pred = infer_using_KspaceNet(image_fft_r,image_fft_i,LayerNum,model=model2,mask=mask,mask_c=mask_c,t_c=1,t_p=4,PosEncoding=PosEncoding.to(device))

        bad_imgs  = torch.concat((T1_pred,T2_pred),dim=1).cuda()

        good_imgs_T1 = image[:,1,:,:][:,np.newaxis,:,:].cuda()
        good_imgs_T2 = image[:,4,:,:][:,np.newaxis,:,:].cuda()

        fakeT1 = netG1(bad_imgs.float()).to(device) + bad_imgs[:,0,:,:][:,np.newaxis,:,:]
        fakeT2 = netG2(bad_imgs.float()).to(device) + bad_imgs[:,1,:,:][:,np.newaxis,:,:]

        errG_fft, errG_mse, fakeT1 = cal_losses(fakeT1,good_imgs_T1,1)
        errG_fft, errG_mse, fakeT2 = cal_losses(fakeT2,good_imgs_T2,4)

        save_slices(fakeT1,good_imgs_T1,fakeT2,good_imgs_T2,label=image[:,6,:,:],subjectNum=subjectNum,layerNum=LayerNum,root=save_to,percent=percent)

            