import torch
from torch import nn
import dataloader as dl
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import os
from tqdm import tqdm
import models
import utils
import torch.optim as optim
import numpy as np
import torchvision.utils as vutils
from scipy.io import savemat


kmodel_dir1 = '/content/mriRecon/trainedModels/modelt1.pt'
kmodel_dir2 = '/content/mriRecon/trainedModels/modelt2.pt'
save_to_dir = '/content/drive/MyDrive/trainedModels/Unet/T2/20/'

mask = utils.load_mask('GaussianDistribution1DMask_10_257')
# mask = load_mask('UniformDistribution1DMask_30_257')
# mask = load_mask('Deterministic1DMask_40_257')
mask_c = 1 - mask

ngpu = 1
lr = 0.0001
lr_decay = 0.5
decay_every = 5
n_epoch = 6
beta1 = 0.5

# Lists to keep track of progress
img_list = []
G_losses = []
G_mse_losses = []
G_fft_mse_losses = []
val_G_losses_mean = []
val_G_mse_losses_mean = []
val_G_fft_losses_mean = []

PSNR_V = []
SSIM_V = []
MSE_FFT_V = []
MSE_IMG_V = []
iter_V = []

PSNR = []
SSIM = []
MSE_FFT = []
MSE_IMG = []
iter_T = []



torch.cuda.empty_cache()

batch_size = 10
dataset = dl.Brats2013_2D(root=dl.train_root, PE= dl.PosEncoding)
dataset_val = dl.Brats2013_2D(root=dl.validation_root , PE= dl.PosEncoding)
dataloader = dl.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False,drop_last=True)
dataloader_val = dl.DataLoader(dataset_val, batch_size=batch_size, shuffle=True,pin_memory=False,drop_last=True)


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

Pred_slice = 4
Pred_slice_feed = 1
image_fft_r, image_fft_i, image, LayerNum, PosEncoding  = next(iter(dataloader_val))

mask = torch.tensor(mask).to(device)
mask_c = torch.tensor(mask_c).to(device)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

print('netG to device..')
netG = models.u_net_bn(ngpu).to(device)
netG = netG.float()
## Define the model
model2 = models.KspaceNetT1(ngpu).to(device)
model1 = models.KspaceNetT1(ngpu).to(device)

print('netG weights init..')
netG.apply(weights_init)

criterion_mse = torch.nn.MSELoss()

# Setup Adam optimizers
print('Setup Adam optimizers for G')
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
scheduler2 = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.5)

print(f'scheduler : {scheduler2.get_last_lr()}')
## Define Metrics
psnr = PeakSignalNoiseRatio().to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

## Load the model if exists, pre-trained model
if  os.path.isfile(kmodel_dir2 + 'modelt2.pt'):
    model2 = torch.load(kmodel_dir2 + 'modelt2.pt')
    
model2.eval()
    
    

if  os.path.isfile(kmodel_dir1 + 'modelt1.pt'):
    model1 = torch.load(kmodel_dir1 + 'modelt1.pt')
    
model1.eval()



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

def cal_psnr_ssim(fake,good_imgs,batch_size):
    PSNR_t = 0 
    SSIM_t = 0
    for ii in range(fake.shape[0]):
        PSNR_t = psnr(fake[ii,:,:,:][np.newaxis,:,:,:], good_imgs[ii,:,:,:][np.newaxis,:,:,:]).cpu().item() + PSNR_t
        SSIM_t = ssim(fake[ii,:,:,:][np.newaxis,:,:,:], good_imgs[ii,:,:,:][np.newaxis,:,:,:]).cpu().item() + SSIM_t
    return PSNR_t/batch_size , SSIM_t/batch_size

T1_pred = infer_using_KspaceNet(image_fft_r,image_fft_i,LayerNum,model=model1,mask=mask,mask_c=mask_c,t_c=4,t_p=1,PosEncoding=PosEncoding.to(device))
T2_pred = infer_using_KspaceNet(image_fft_r,image_fft_i,LayerNum,model=model2,mask=mask,mask_c=mask_c,t_c=1,t_p=4,PosEncoding=PosEncoding.to(device))
utils.imshow(vutils.make_grid((T1_pred).detach().cpu(), padding=2, nrow=5, normalize=True), batch_size, 'T1_pred', cmap = 'gray')
utils.imshow(vutils.make_grid((T2_pred).detach().cpu(), padding=2, nrow=5, normalize=True), batch_size, 'T2_pred', cmap = 'gray')
fix_bad_imgs  = torch.concat((T1_pred,T2_pred),dim=1).cuda()
fix_good_imgs = image[:,Pred_slice,:,:][:,np.newaxis,:,:].cuda()
psnrValue , ssimValue = cal_psnr_ssim(fix_bad_imgs[:,1,:,:][:,np.newaxis,:,:],fix_good_imgs,batch_size)
print(psnrValue)
print(ssimValue)
clip = torch.ones_like(fix_good_imgs).to(device) * 0.2
utils.imshow(vutils.make_grid(torch.minimum(clip,torch.abs(fix_good_imgs-T1_pred)).detach().cpu(), padding=2, nrow=5, normalize=True), batch_size, 'T1_pred_diff', cmap = 'jet')
utils.imshow(vutils.make_grid(fix_good_imgs.detach().cpu(), padding=2, nrow=5, normalize=True), batch_size, 'ground_truth', cmap = 'gray')
import time
intialTime = time.time()
flag = True
t_iters = 0

for epoch in range(n_epoch):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        netG.train(True)
        image_fft_r, image_fft_i, image, LayerNum, PosEncoding = data
        image = image.to(device)
        image_fft_i = image_fft_i.to(device)
        image_fft_r = image_fft_r.to(device)
        model1.train(False)
        model2.train(False)
        
        T1_pred = infer_using_KspaceNet(image_fft_r,image_fft_i,LayerNum,model=model1,mask=mask,mask_c=mask_c,t_c=4,t_p=1,PosEncoding=PosEncoding.to(device))
        T2_pred = infer_using_KspaceNet(image_fft_r,image_fft_i,LayerNum,model=model2,mask=mask,mask_c=mask_c,t_c=1,t_p=4,PosEncoding=PosEncoding.to(device))

        netG.zero_grad()

        bad_imgs  = torch.concat((T1_pred,T2_pred),dim=1).cuda()
        good_imgs = image[:,Pred_slice,:,:][:,np.newaxis,:,:].cuda()
    
        fake = netG(bad_imgs.float()).to(device) + bad_imgs[:,Pred_slice_feed,:,:][:,np.newaxis,:,:]
       
        errG_fft, errG_mse, fake= cal_losses(fake,good_imgs,Pred_slice)
        errG =  errG_mse + errG_fft

        # Calculate gradients for G
        errG.backward()
        # Update G
        optimizerG.step()

        netG.train(False)
        with torch.no_grad():
            if i % 10 == 0: 
                MSE_IMG.append(errG_mse.item())
                MSE_FFT.append(errG_fft.item())
                psnrValue , ssimValue = cal_psnr_ssim(fake,good_imgs,batch_size)
                PSNR.append(psnrValue)
                SSIM.append(ssimValue)
                iter_T.append(t_iters)
                print('Elapsed time = {}'.format(time.time() - intialTime))
                intialTime = time.time()
                # Output training stats
                print('[%d/%d][%d/%d]\tLoss_G: %.4f, rrG_mse: %.4f, errG_fft_mse: %.4f, SSIM: %.4f, PSNR: %.4f' % (epoch, n_epoch, i,
                 len(dataloader), errG.item(), MSE_IMG[-1], MSE_FFT[-1], SSIM[-1], PSNR[-1]))
                # utils.imshow(vutils.make_grid((fake).detach().cpu(), padding=2, nrow=5, normalize=True),
                #                                          batch_size, '{}_{}_fix'.format(epoch, iter))
            
                mdic = {"PSNR": PSNR,"SSIM": SSIM,"MSE_FFT": MSE_FFT,"MSE_IMG": MSE_IMG,'iter_T':iter_T
                       ,"PSNR_V": PSNR_V,"SSIM_V": SSIM_V,"MSE_FFT_V": MSE_FFT_V,"MSE_IMG_V": MSE_IMG_V,'iter_V':iter_V}
                savemat(save_to_dir + "Training_Losses.mat", mdic)

                # Check how the generator is doing by saving G's output on fixed_noise
                if (i % 500 == 0) or ((epoch == n_epoch - 1) and (i == len(dataloader) - 1)) or ((PSNR[-1] > np.max(PSNR[0:-1])) and t_iters > 199):
                    print('calculating metrics and loss for validation dataset...')
                    PSNR_Vt = 0
                    SSIM_Vt = 0
                    MSE_FFT_Vt = 0
                    MSE_IMG_Vt = 0

                    for val_iter, data in tqdm(enumerate(dataloader_val,0)):
        
                        image_fft_r, image_fft_i, image, LayerNum, PosEncoding = data
                        image = image.to(device)
                        image_fft_i = image_fft_i.to(device)
                        image_fft_r = image_fft_r.to(device)
                        model1.train(False)
                        model2.train(False)
                        T1_pred = infer_using_KspaceNet(image_fft_r,image_fft_i,LayerNum,model=model1,mask=mask,mask_c=mask_c,t_c=4,t_p=1,PosEncoding=PosEncoding.to(device))
                        T2_pred = infer_using_KspaceNet(image_fft_r,image_fft_i,LayerNum,model=model2,mask=mask,mask_c=mask_c,t_c=1,t_p=4,PosEncoding=PosEncoding.to(device))
                        if flag == True:
                            utils.imshow(vutils.make_grid((T1_pred).detach().cpu(), padding=2, nrow=5, normalize=True), batch_size, 'T1_pred', cmap = 'gray')
                            utils.imshow(vutils.make_grid((T2_pred).detach().cpu(), padding=2, nrow=5, normalize=True), batch_size, 'T2_pred', cmap = 'gray')
                            flag = False

                        # netG.zero_grad()

                        bad_imgs  = torch.concat((T1_pred,T2_pred),dim=1).cuda()
                        good_imgs = image[:,Pred_slice,:,:][:,np.newaxis,:,:].cuda()
                    
                        fake = netG(bad_imgs.float()).to(device) + bad_imgs[:,Pred_slice_feed,:,:][:,np.newaxis,:,:]
                        
                        errG_fft, errG_mse, fake = cal_losses(fake,good_imgs,Pred_slice)
                        errG =  errG_mse + errG_fft

                        psnrValue , ssimValue = cal_psnr_ssim(fake,good_imgs,batch_size)

                        PSNR_Vt = PSNR_Vt + psnrValue
                        SSIM_Vt = SSIM_Vt + ssimValue
                        MSE_FFT_Vt = MSE_FFT_Vt + errG_mse.item()
                        MSE_IMG_Vt = MSE_IMG_Vt + errG_fft.item()
                    
                    normalizing_factor = len(dataloader_val)

                    PSNR_V.append(PSNR_Vt/normalizing_factor)
                    SSIM_V.append(SSIM_Vt/normalizing_factor)
                    MSE_FFT_V.append(MSE_FFT_Vt/normalizing_factor)
                    MSE_IMG_V.append(MSE_IMG_Vt/normalizing_factor)
                    iter_V.append(t_iters)
                    mdic = {"PSNR": PSNR,"SSIM": SSIM,"MSE_FFT": MSE_FFT,"MSE_IMG": MSE_IMG,'iter_T':iter_T
                       ,"PSNR_V": PSNR_V,"SSIM_V": SSIM_V,"MSE_FFT_V": MSE_FFT_V,"MSE_IMG_V": MSE_IMG_V,'iter_V':iter_V}
                    savemat(save_to_dir + "Training_Losses.mat", mdic)
                    fake = netG(fix_bad_imgs.float()).to(device) + fix_bad_imgs[:,Pred_slice_feed,:,:][:,np.newaxis,:,:]
                    errG_fft, errG_mse, fake= cal_losses(fake,fix_good_imgs,Pred_slice)
                    clip = torch.ones_like(fake).to(device) * 0.2
                    print(f'sum of error: {torch.sum(torch.abs(fix_good_imgs-fake))}')
                    print(f'psnr of fixed: { psnr(fake, fix_good_imgs).cpu()}')
                              
                    if len(PSNR_V) > 1:
                        print(f"max:{np.max(PSNR_V[0:-1])},current:{PSNR_Vt/normalizing_factor}")
                        if PSNR_Vt/normalizing_factor > np.max(PSNR_V[0:-1]):
                            print('new best model found!!saving...')
                            torch.save(netG, save_to_dir + 'model_unet.pt')
                                    
        t_iters += 1
    scheduler2.step()
    print(scheduler2.get_last_lr())