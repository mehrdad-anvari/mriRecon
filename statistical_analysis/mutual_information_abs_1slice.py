import numpy as np
from medpy.io import load, save
from sklearn.feature_selection import mutual_info_regression as MIC

fft_abs_path = 'K:\\academic\\Datasets\\BrainMRIstatistics\\70slices_fft_abs.mha'

fft_abs, header = load(fft_abs_path)

N = 2
mutual_information = np.zeros((240, 240, N))
k=0
for i1 in [25,115]:
    # for j1 in range(N):
    j1 = i1
    print(f'i1 = {i1},j1 = {j1}')
    for i2 in range(240):
        for j2 in range(240):
            fft_signal1 =(fft_abs[i1, j1, :].reshape(-1))
            fft_signal2 =(fft_abs[i2, j2, :].reshape(-1,1))
            # fft_signal1 = fft_signal1/np.mean(fft_signal1)
            # fft_signal2 = fft_signal2/np.mean(fft_signal2)
            mutual_information[i2, j2, k] = MIC(fft_signal2, fft_signal1)[0]
        if i2 % 10 == 0:
            print(i2)
    k = k+1
save(mutual_information, 'new_mutual_info_abs_1slice_v2.mha')