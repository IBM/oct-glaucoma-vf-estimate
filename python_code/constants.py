import numpy as np

def get_vft_mask():
    arr = np.zeros((9, 9), np.int)
    arr[0, 3:7] = 1
    arr[7, 3:7] = 1

    arr[1, 2:8] = 1
    arr[6, 2:8] = 1

    arr[2, 1:] = 1
    arr[5, 1:] = 1


    arr[3:5, :] = 1
    arr[3, -2] = 0
    arr[4, -2] = 0
    return arr
VFT_Mask = get_vft_mask()
VFT_range = [0, 35.0]
bin_size =4
VFI_range = [0, 100]  # 0,100
MD_range = [-32, 5]  # - 31.97,5.02
PSD_range = [1, 16]  # 0.81 ,17.09
bins = np.arange(VFT_range[0]+1, VFT_range[1] + bin_size, bin_size)
#print(bins) # [ 1.  6. 11. 16. 21. 26. 31. 36.]

############Network parameters #############
dpth = 128
SHAPE=112
BATCH=2
nfilters = [16,32,32]
nfilters_merged = [64]

# digitize_VFT --> True: Quantize, False: Normalize
# CNN_OUT_GarWayHeathmap -->True: 10 flatten values, False for VFT threshold values
###############EXP1:VFT THRESHOLD PREDICTION Quantization ##################
#digitize_VFT = True; CNN_OUT_GarWayHeathmap = False; vft_shape=(9, 9,10)

###############EXP2:VFT THRESHOLD PREDICTION Normalization ##################
digitize_VFT = False; CNN_OUT_GarWayHeathmap = False ; vft_shape=(9, 9)

###############EXP3:VFT Heathmap REGIONS PREDICTION (10) ##################
#digitize_VFT = False; CNN_OUT_GarWayHeathmap = True; vft_shape=(9, 9)
out_num=3 # or 10 for garway heathmap values
