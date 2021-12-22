import cv2
import random
from PIL import Image
import pickle
import random
from glob import glob
from tensorpack.utils.gpu import get_num_gpu
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from tensorpack import *
import pandas as pd
from vftparser import *
import scipy as sp

class VFTThicknessDataFlow(DataFlow):
    def __init__(self, maps_data_dir='', vft_data_dir='', task='test', Thicknesses=['RNFL', 'GCIPL'], fold=1,vft_type = 'THRESHOLD'):

        #Thicknesses = ['RNFL', 'GCIPL', 'GCIPL_wd', 'RNFL_wd']
        self.vft_type = vft_type
        self.VFT_col = 'Name'
        self.Fold_col = 'Fold_' + str(fold)
        df = pd.read_csv('oct_onh_mac_vft_data_13119_folds.csv', index_col=0)
        cond = (df['TEST_PATTERN'] == 25) & (df['TEST_STRATEGY'] == 4)

        df = df[cond]

        self.Thicknesses_names = Thicknesses
        self.OCT_col = 'P_onh'
        self.uid_col = 'Name_onh'
        self.garway_map, self.sectors_info = get_garway_heathmap()


        if task == 'all':
            self.filelist = df
        elif task == 'test':  # test all files
            self.filelist = df[df[self.Fold_col] == 'test']  # .iloc[:500,:]
        elif task == 'train':
            self.filelist = df[df[self.Fold_col] == 'train']  # .iloc[:2000,:]
        elif task == 'val':
            self.filelist = df[df[self.Fold_col] == 'val']  # .iloc[:500,:]
        print('*' * 100)
        print('{} Data Size: {}'.format(task.capitalize(), len(self.filelist)))
        print('*' * 100)
        self.maps_data_dir = maps_data_dir
        self.vft_data_dir = vft_data_dir

    def __iter__(self):
        for s in range(len(self.filelist)):
            row = self.filelist.iloc[s, :]  # .to_list()
            imgs = self.preprocess_thickness_maps(row[self.OCT_col], self.maps_data_dir)
            sectors, global_vft, local_vft = self.process_vft(row[self.VFT_col], self.vft_data_dir)
            # print(local_vft.shape)
            if out_num == 3:
                yield imgs + [global_vft, local_vft, row[self.uid_col]]
            else:  # with Garway Heathmap sectors
                yield imgs + [np.concatenate((sectors, global_vft)), local_vft,
                              row[self.uid_col]]  # local_vft, sectors,global_vft,,row['uid_onh']]  # , fov_mask]
            # yield [img1, img2, img3, img4, local_vft, sectors,global_vft,row['uid_onh']]  # , fov_mask]

    def __len__(self):
        return len(self.filelist)

    def process_vft(self, fname, data_dir):
        filepath = data_dir + fname + '.xml'
        sectors, global_vft, local_vft = read_vft_heathmap(filepath, vkind=self.vft_type, # 'THRESHOLD'
                                                           garway_map=self.garway_map,
                                                           heathmap_sectors=self.sectors_info)

        return sectors, global_vft, local_vft

    def preprocess_thickness_maps(self, fname,data_dir):
        fname = fname.split('/')[-1].split('.')[0]
        i =0
        im = []
        for layer_name in self.Thicknesses_names:
            fpath = data_dir +layer_name+'/'+fname + '_seg'+layer_name+'_thickness.png'

            im.append(cv2.imread(fpath,cv2.IMREAD_GRAYSCALE))

            '''
            
            if i  == 0 :
                im = np.expand_dims(cv2.imread(fpath,cv2.IMREAD_GRAYSCALE),axis=-1)
                i +=1
            else:
                im1 = np.expand_dims(cv2.imread(fpath,cv2.IMREAD_GRAYSCALE),axis=-1)
                im = np.concatenate((im1, im), axis=-1)
            '''
        return im


def get_data( mapsdatadir,vftdatadir, SHAPE=128, BATCH=4, task='test', Thicknesses=['RNFL_wd', 'GCIPL_wd'], fold=1, vft_type = 'THRESHOLD'):
    if task == 'train':
        #augs = [imgaug.Resize(SHAPE)]

        augs =[
            imgaug.Resize(int(SHAPE * 1.12)),
            imgaug.RandomCrop(SHAPE),
            imgaug.Flip(horiz=True),
            #imgaug.Rotation(15)
        ]

    else:
        augs = [imgaug.Resize(SHAPE)]

    def get_image_pairs(mapsdatadir, vftdatadir):
        def get_df(mapsdatadir, vftdatadir):
            df = VFTThicknessDataFlow(maps_data_dir = mapsdatadir, vft_data_dir=vftdatadir, task=task, Thicknesses =Thicknesses, fold=fold, vft_type = vft_type)
            return AugmentImageComponents(df, augs, index=[0,1])

        return get_df(mapsdatadir, vftdatadir)

    df = get_image_pairs(mapsdatadir, vftdatadir)
    df = BatchData(df, BATCH, remainder=False if task == 'train' else True)
    size = df.size()
    # if isTrain:
    #    df = PrefetchData(df, num_prefetch=1, num_proc=1)

    # df = PrefetchDataZMQ(df, 2 if isTrain else 1)#

    #df = MultiProcessRunnerZMQ(df, get_num_gpu() if task == 'train' else 1)

    return df, size


if __name__ == "__main__":


    mapsdatadir, vftdatadir = '/Users/gyasmeen/Downloads/Thickness_Maps/' , '/Users/gyasmeen/Desktop/Results/nyu_vft_xml/'
    #df = VFTThicknessDataFlow(maps_data_dir=mapsdatadir, vft_data_dir=vftdatadir, task='train')

    df,batch_num = get_data('/Users/gyasmeen/Downloads/Thickness_Maps/', '/Users/gyasmeen/Desktop/Results/nyu_vft_xml/',task = 'train')
    ### check if augmentation can  happen on 3d data
    df.reset_state()
    c = 1

    for batch in df:
        print(len(batch[0]))
        img1, img2, sectors_global_vft, local_vft,uid = batch
        img1, img2, sectors_global_vft, local_vft, uid = img1[0], img2[0],sectors_global_vft[0] ,local_vft[0],uid[0]


        # take the first sample in a given batch
        print(local_vft.min(), local_vft.max())

        fig, axes = plt.subplots(2, 3, figsize=(20, 20))
        axes = axes.flatten()

        i = 0
        for img in [img1, img2, local_vft]:
            img = np.squeeze(np.sum(img, axis=2)) if len(img.shape) > 2 else img
            ax = axes[i];
            i += 1
            ax.imshow(img, interpolation='nearest', cmap='gray', vmax=img.max(), vmin=img.min())
            ax.axis('off')
        plt.title(uid)
        plt.savefig('SampleImages' + str(c) + '.png', bbox_inches='tight')
        plt.show()
        c += 1
        if c==3:
            break
