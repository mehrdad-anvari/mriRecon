import numpy as np
from medpy.io import load, save
from sklearn.feature_selection import mutual_info_regression as MIC

fft_abs_path = 'K:\\academic\\Datasets\\BrainMRIstatistics\\70_150_slices_fft_abs.mha'

fft_abs, header = load(fft_abs_path)

N= 240               
mutual_information = np.zeros((N, N, 2))
for k in range(1,3):
    print(f' k = {k}')
    for i1 in range(N):
        print(i1)
        for j1 in range(N):
            fft_signal1 = fft_abs[i1, j1, :, 0].reshape(-1,1)
            fft_signal2 = fft_abs[i1, j1, :, k].reshape(-1)  
            mutual_information[i1, j1, k-1] = MIC(fft_signal1, fft_signal2)[0]

save(mutual_information, 'new_mutual_info_abs_2slices.mha')
