from medpy.io import load, save
import numpy as np
import matplotlib.pyplot as plt
import glob

save_dir = 'J:/Anvari/preprocessedDataset2/'
ls_files_T1 = glob.glob('J:/Anvari/data/BRATS2015_Training/*/*/*MR_T1.*/*.mha')
ls_files_OT = glob.glob('J:/Anvari/data/BRATS2015_Training/*/*/*OT.*/*.mha')

dataUnit = np.zeros((257,257,11))

for i in range(len(ls_files_OT)):
    
    img_T1,header = load(ls_files_T1[i])
    img_OT,header = load(ls_files_OT[i])

    img_T1 = img_T1/np.max(img_T1)

    img_T1 = np.pad(img_T1,((8,9),(8,9),(0,0)))
    img_OT = np.pad(img_OT,((8,9),(8,9),(0,0)))

    for j in range(2,img_T1.shape[2]-2):
        nonZeroRatio = np.sum((img_OT[:,:,j]>0).flatten())
        # print(nonZeroRatio)
        if nonZeroRatio > 100:
            dataUnit[:,:,0] = img_T1[:,:,j-1]
            dataUnit[:,:,1] = img_T1[:,:,j]
            dataUnit[:,:,2] = img_T1[:,:,j+1]

            dataUnit[:,:,3] = img_OT[:,:,j]

            save(dataUnit, save_dir + f'{i}_{j}.mha', use_compression = True)