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

'''
['subject_id_onh', 'age_at_visit_date_onh', 'visit_date_onh', 'eye_onh',
       'scan_time_onh', 'scan_type_onh', 'ss_onh', 'avgthickness_onh',
       'clockhour1', 'clockhour2', 'clockhour3', 'clockhour4', 'clockhour5',
       'clockhour6', 'clockhour7', 'clockhour8', 'clockhour9', 'clockhour10',
       'clockhour11', 'clockhour12'
       ,'quad_t', 'quad_s', 'quad_n', 'quad_i', 'symmetry', 'rimarea',
       'discarea', 'avg_cd_ratio', 'vert_cd_ratio', 'cupvol', 'qualified_onh',
       'uid_onh', 'uid_short_onh', 'Name_onh', 'P_onh', 'subject_id_mac',
       'age_at_visit_date_mac', 'visit_date_mac', 'eye_mac', 'scan_time_mac'
       ,'scan_type_mac', 'ss_mac', 'center_mac', 'inn_nasal', 'inn_sup',
       'inn_temp', 'inn_inf', 'out_nasal', 'out_sup', 'out_temp', 'out_inf',
       'gca_average', 'gca_tempsup', 'gca_sup', 'gca_nassup', 'gca_nasinf',
       'gca_inf', 'gca_tempinf', 'rnfl_average', 'rnfl_tempsup', 'rnfl_sup',
       'rnfl_nassup', 'rnfl_nasinf', 'rnfl_inf', 'rnfl_tempinf', 'or_average',
       'or_tempsup', 'or_sup', 'or_nassup', 'or_nasinf', 'or_inf',
       'or_tempinf', 'qualified_mac', 'uid_mac', 'uid_short_mac', 'Name_mac',
       'P_mac', 'TEST_PATTERN', 'TEST_STRATEGY', 'STIMULUS_COLOR',
       'STIMULUS_SIZE', 'BACKGROUND_COLOR', 'EXAM_DURATION', 'FIXATION_TARGET',
       'FIXATION_MONITOR', 'BLIND_SPOT_X', 'BLIND_SPOT_Y',
       'BLIND_SPOT_STIMULUS_SIZE', 'FALSE_NEGATIVE_METHOD',
       'FALSE_NEGATIVE_PERCENT', 'FALSE_POSITIVE_METHOD',
       'FALSE_POSITIVE_PERCENT', 'TRIALS', 'ERRORS', 'FOVEAL_RESULT',
       'FOVEAL_THRESHOLD', 'CENTRAL_REF_LEVEL', 'THROWN_OUT_POINTS',
       'MINIMUM_STIMULUS', 'FIELD_SIZE', 'LANGUAGE', 'SF_STATUS',
       'NUM_THRESHOLD_POINTS', 'DISPLAY_NAME', 'VISIT_DATE',
       'SERIES_DATE_TIME', 'PUPIL_DIAMETER', 'PUPIL_DIAMETER_AUTO',
       'EXAM_TIME', 'Name', 'MD', 'MD_PROBABILITY', 'PSD', 'PSD_PROBABILITY',
       'VFI', 'CPSD', 'CPSD_PROBABILITY', 'SF_PROBABILITY', 'GHT', 'uid',
       'subject_id', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5',
       'Fold_6', 'Fold_7', 'Fold_8', 'Fold_9', 'Fold_10']
'''

class VFTOCTDataFlow(DataFlow):
    def __init__(self, oct_data_dir='',vft_data_dir='',task='test',Multi_Input=True,OCT_TYPE='onh',fold =1,vft_type = 'THRESHOLD'):

        self.DEBUG = False
        self.Multi_Input = Multi_Input
        self.vft_type= vft_type
        if OCT_TYPE =='onh':
            self.OCT_col = 'P_onh'
            self.uid_col = 'Name_onh'
        else:
            self.OCT_col = 'P_mac'
            self.uid_col = 'Name_mac'


        self.VFT_col = 'Name'
        # 'P_mac' --> mac path
        # 'P_onh' --> onh path
        # 'Name' --> xml file name
        self.Fold_col = 'Fold_'+ str(fold)
        df = pd.read_csv('oct_onh_mac_vft_data_13119_folds.csv', index_col=0)
        cond  = (df['TEST_PATTERN']==25) & (df['TEST_STRATEGY']==4)

        df =df[cond]
        self.garway_map, self.sectors_info = get_garway_heathmap()

        if task == 'all':

            '''
            
            # JUPYTEER DESKTOP TEST
            files_list = glob('/Users/gyasmeen/Desktop/Results/sample_oct/*.img')
        
            df['filename'] = df[self.OCT_col].apply(lambda x: x.split('/')[-1])
            sample_df = pd.DataFrame()
            for f in files_list:
                s = df[df['filename'] == f.split('/')[-1]]
                if len(s) == 1:
                    sample_df = sample_df.append(s)
            self.filelist = sample_df
            '''



            self.filelist = df

        elif task == 'test': # test all files
            self.filelist = df[df[self.Fold_col] == 'test']#.iloc[:500,:]
        elif task == 'train':
            self.filelist = df[df[self.Fold_col] == 'train']#.iloc[:2000,:]
        elif task == 'val':
            self.filelist = df[df[self.Fold_col] == 'val']#.iloc[:500,:]
        print('*' * 100)
        print('{} Data Size: {}'.format(task.capitalize() ,len(self.filelist)))
        print('*' * 100)
        self.oct_data_dir = oct_data_dir
        self.vft_data_dir = vft_data_dir
    def __iter__(self):
        for s in range(len(self.filelist)):
            row = self.filelist.iloc[s, :]  # .to_list()
            imgs = self.preprocess_oct(row[self.OCT_col], self.oct_data_dir)
            sectors, global_vft, local_vft = self.process_vft(row[self.VFT_col],self.vft_data_dir)
            #print(local_vft.shape)
            if out_num ==3:
                yield imgs + [global_vft,local_vft, row[self.uid_col]]
            else: # with Garway Heathmap sectors
                yield imgs+[np.concatenate((sectors, global_vft)),local_vft, row[self.uid_col]]#local_vft, sectors,global_vft,,row['uid_onh']]  # , fov_mask]
            #yield [img1, img2, img3, img4, local_vft, sectors,global_vft,row['uid_onh']]  # , fov_mask]

    def __len__(self):
        return len(self.filelist)

    def process_vft(self, fname, data_dir):
        filepath = data_dir+fname+'.xml'
        sectors, global_vft, local_vft = read_vft_heathmap(filepath, vkind=self.vft_type, # 'PATTERN' , 'THRESHOLD'
                                                                    garway_map=self.garway_map,
                                                                    heathmap_sectors=self.sectors_info)

        return sectors, global_vft, local_vft
    def resize_cube(self,cube):
        """Return resized cube with the define shape"""
        zoom = [float(x) / y for x, y in zip((SHAPE,SHAPE,dpth), cube.shape)]
        resized = sp.ndimage.zoom(cube, zoom,mode='nearest')


        #assert resized.shape == shape
        return resized
    def preprocess_oct(self,fname,data_dir):

        fpath = data_dir + fname.split('/')[-1]
        cube  = self.read_single_oct(fpath)

        # split it into 4 parts based on depth
        if self.Multi_Input:
            return [cube[:, :, 0:256:2], cube[:, :, 256:512:2], cube[:, :, 512:768:2], cube[:, :, 768:1024:2]]
        else:
            return [self.resize_cube(cube)]

    def read_single_oct(self, fpath):
        with open(fpath, 'rb') as f:
            data = np.frombuffer(f.read(), 'uint8')
            cube = data.reshape((200, 1024, 200), order='C')
            if 'OS' in fpath:
                cube = cube[:, ::-1, ::-1]
        cube = np.transpose(cube, (0, 2, 1))
        #im_enface = np.squeeze(np.average(cube.astype(np.float32), axis=1))
        #print(cube.shape) # 200x200x1024
        return cube

def get_data(octdatadir,vftdatadir, SHAPE=128,BATCH=4 , task='test',Multi_Input=True,OCT_TYPE='onh',fold = 1,vft_type = 'THRESHOLD'):
    if task=='train':
        augs = [imgaug.Resize(SHAPE)]
        '''
        augs =[
            imgaug.Resize(int(SHAPE * 1.12)),
            imgaug.RandomCrop(SHAPE),
            imgaug.Flip(horiz=True),
            #imgaug.Rotation(15)
        ]
        '''
    else:
        augs = [imgaug.Resize(SHAPE)]

    def get_image_pairs(octdatadir,vftdatadir):
        def get_df(octdatadir,vftdatadir):
            df = VFTOCTDataFlow(oct_data_dir=octdatadir,vft_data_dir=vftdatadir, task=task,Multi_Input=Multi_Input,OCT_TYPE=OCT_TYPE,fold = fold,vft_type=vft_type)
            return AugmentImageComponents(df, augs, index=(0,1,2,3)) if Multi_Input else df
        return get_df(octdatadir,vftdatadir)

    df = get_image_pairs(octdatadir,vftdatadir)
    df = BatchData(df, BATCH, remainder=False if task == 'train' else True)
    size=df.size()
    #if isTrain:
    #    df = PrefetchData(df, num_prefetch=1, num_proc=1)

    #df = PrefetchDataZMQ(df, 2 if isTrain else 1)#

    df = MultiProcessRunnerZMQ(df, get_num_gpu() if task == 'train' else 1)

    return df,size

if __name__ == "__main__":

    # 3D shape
    base_dir ='/Users/gyasmeen/Desktop/Results/Reconstruction/notebooks/hvfte_preprocessing/sample_data/'
    #df = VFTOCTDataFlow(oct_data_dir = base_dir+'oct/',vft_data_dir='/Users/gyasmeen/Desktop/Results/nyu_vft_xml/' ,task='train')
    df,batch_num = get_data(base_dir+'oct/', '/Users/gyasmeen/Desktop/Results/nyu_vft_xml/', task='train')

    df.reset_state()
    c = 1

    for batch in df:
        print(len(batch[0]))
        img1, img2, img3, img4, sectors_global_vft, local_vft,uid = batch
        img1, img2, img3, img4, sectors_global_vft, local_vft, uid = img1[0], img2[0], img3[0], img4[0],sectors_global_vft[0] ,local_vft[0],uid[0]

        # take the first sample in a given batch
        print(local_vft.min(), local_vft.max())
        #print(sectors_global_vft)

        #print(uid)
        #print(img1.shape,img2.shape, img3.shape,img4.shape) #(128, 128, 256) (128, 128, 256) (128, 128, 256) (128, 128, 256)
        #print(img1.dtype, img2.dtype, img3.dtype, img4.dtype) #uint8 uint8 uint8 uint8
        #print(sectors.shape,global_vft.shape,local_vft.shape) #(7,) (3,) (8, 9)

        fig, axes = plt.subplots(2, 3, figsize=(10, 10))
        axes = axes.flatten()

        i=0
        for img in [img1, img2, img3, img4, local_vft]:

            img = np.squeeze(np.sum(img, axis=2)) if len(img.shape) >2 else img
            print(img.shape)
            ax = axes[i]; i+=1
            ax.imshow(img,interpolation='nearest', cmap='gray', vmax=img.max(), vmin=img.min())
            ax.axis('off')
        plt.title(uid)
        plt.savefig('SampleImages'+str(c)+'.png', bbox_inches='tight')
        plt.show()
        c += 1
        break
