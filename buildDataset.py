from medpy.io import load, save
import numpy as np
import matplotlib.pyplot as plt
import glob


save_dir = '/content/BRATS2015_Training_Processed'
ls_files_T1 = glob.glob('/content/BRATS2015_Training/*/*/*MR_T1.*/*.mha')
ls_files_T2 = glob.glob('/content/BRATS2015_Training/*/*/*MR_T2.*/*.mha')
ls_files_OT = glob.glob('/content/BRATS2015_Training/*/*/*OT.*/*.mha')

dataUnit = np.zeros((257,257,7))

for i in range(len(ls_files_OT)):
    if i%10 == 0:
      print(f'{i}/{len(ls_files_OT)}')
    
    img_T1,header = load(ls_files_T1[i])
    img_T2,header = load(ls_files_T2[i])
    img_OT,header = load(ls_files_OT[i])

    img_T1 = img_T1/np.max(img_T1)
    img_T2 = img_T2/np.max(img_T2)

    img_T1 = np.pad(img_T1,((8,9),(8,9),(0,0)))
    img_T2 = np.pad(img_T2,((8,9),(8,9),(0,0)))
    img_OT = np.pad(img_OT,((8,9),(8,9),(0,0)))

    for j in range(2,img_T1.shape[2]-2):
        nonZeroRatio = np.sum((img_OT[:,:,j]>0).flatten())
        # print(nonZeroRatio)
        if nonZeroRatio > 100:
            dataUnit[:,:,0] = img_T1[:,:,j-1]
            dataUnit[:,:,1] = img_T1[:,:,j]
            dataUnit[:,:,2] = img_T1[:,:,j+1]

            dataUnit[:,:,3] = img_T2[:,:,j-1]
            dataUnit[:,:,4] = img_T2[:,:,j]
            dataUnit[:,:,5] = img_T2[:,:,j+1]

            dataUnit[:,:,6] = img_OT[:,:,j]

            if i < 265:
                save(dataUnit, save_dir + '/train/' + f'{i}_{j}.mha', use_compression = True)
            else:
                save(dataUnit, save_dir + '/test/'  + f'{i}_{j}.mha', use_compression = True)
