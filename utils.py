import matplotlib.pyplot as plt
import scipy
import numpy as np
import torch
from medpy.io import save, load

save_to_dir = ''

def load_mask(mask_name):
    dir = 'C:\\Users\\mehrdad\\MS_projects\\mask\\Build mask'
    matfile = scipy.io.loadmat(dir + '\\{}.mat'.format(mask_name)) 
    print(type(matfile))
    print(matfile.keys())
    mask = np.zeros((6,257,257))
    mask_P = matfile.get('PrimaryMask')
    mask_S = matfile.get('SecondaryMask')
    mask_P = np.logical_or(mask_P,np.flip(mask_P))
    mask_S = np.logical_or(mask_S,np.flip(mask_S))
    mask[0,:,:]=mask_S
    mask[1,:,:]=mask_P
    mask[2,:,:]=mask_S
    mask[3,:,:]=mask_P
    mask[4,:,:]=mask_S
    mask[5,:,:]=mask_P
    print(mask.shape)
    print(mask.dtype)
    print(type(mask))
    plt.imshow(mask_P)
    plt.title(mask_name)
    plt.axis('off')
    plt.show()
    plt.imshow(mask_S)
    plt.title(mask_name)
    plt.axis('off')
    plt.show()
    plt.imshow(np.logical_or(mask_P,mask_S))
    plt.title(mask_name)
    plt.axis('off')
    plt.show()
    return mask


def cal_fft_for_all_channels(x):
    fft = torch.zeros_like(x,dtype=torch.complex64)
    fft = torch.fft.fft2(x)
    fft = torch.fft.fftshift(fft,dim=(1,2))
    return torch.real(fft).to(torch.float32) ,torch.imag(fft).to(torch.float32)

def to_bad_img(x, mask):
    x = (x + 1.) / 2.
    for i in range(6):
        fft = scipy.fftpack.fft2(x[i, :, :])
        fft = scipy.fftpack.fftshift(fft)
        fft = fft * mask[i,:,:]
        fft = scipy.fftpack.ifftshift(fft)
        x[:, :, i] = scipy.fftpack.ifft2(fft)
        x[:, :, i] = np.abs(x[:, :, i])
        x[:, :, i] = x[:, :, i] * 2 - 1
    return x

def imshow(img, batch_size, name, cmap):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.figure(figsize=(7, 4))
    if cmap != 'gray':
        # c = 1 / np.log(1 + np.max(npimg[:,:,0]))
        # image = c * (np.log(npimg[:,:,0] + 1))
        image = npimg[:,:,0]
    else:
        image = npimg[:,:,0]

    # cm = plt.get_cmap(cmap)
    # npimg = cm(image)
    if cmap != 'gray':
        image[-1,-1] = 0.2
        image[-2,-1] = 0
        plt.imshow(image,interpolation = 'none',cmap=cmap,vmin=0,vmax=0.2)
        plt.colorbar(orientation="horizontal")
        plt.axis('off')
        plt.savefig(save_to_dir+'{}_02.png'.format(name), format='png', dpi=600)
        plt.figure(figsize=(7, 4))
        image[-1,-1] = 1
        image[-2,-1] = 0
        plt.imshow(image,interpolation = 'none',cmap=cmap,vmin=0,vmax=1)
        plt.colorbar(orientation="horizontal")
        plt.axis('off')
        plt.savefig(save_to_dir+'{}_10.png'.format(name), format='png', dpi=600)
    else:
        image[-1,-1] = 1
        image[-2,-1] = 0
        plt.imshow(image,interpolation = 'none',cmap=cmap,vmin=0,vmax=1)
        plt.axis('off')
        plt.savefig(save_to_dir+'{}.png'.format(name), format='png', dpi=600)
            
def imsave(img,epoch,i,name,batch_size):
    img_slices = np.zeros((6*batch_size,240,240))
    for j in range(batch_size*6):
        img_slices[j,:,:] = img[int(j//6),int(j%6),:,:]
    save(img_slices,save_to_dir+f'{name}_{epoch}_{i}.mha')
