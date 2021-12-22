import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
from tensorpack import *
import sys
import os
import tensorflow as tf
from tensorpack.utils.viz import stack_patches
import cv2
import tensorpack.utils.viz as viz
from tensorpack.tfutils import get_tf_version_tuple
from keras import backend as K
from scipy.stats.stats import pearsonr
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import warnings
warnings.simplefilter("ignore")

def read_single_oct(fpath):
    with open(fpath, 'rb') as f:
        data = np.frombuffer(f.read(), 'uint8')

        cube = data.reshape((200, 1024, 200), order='C')
        if 'OS' in fpath:
            cube = cube[:, ::-1, ::-1]
        # plt.imshow(np.squeeze(np.sum(cube, axis=1)),cmap='gray')
        # plt.show()
    return cube


def plot_training_perf(path):
    pd.options.plotting.backend = "plotly"
    files_list = glob(path)

    final_df = pd.DataFrame()
    for f in files_list:
        df = pd.read_csv(f)

        final_df[f.split('/')[-1].split('-')[-1].split('.')[0]] = df['Value']

    cols = ['mean_squared_error', 'validation_mean_squared_error',
            'mean_absolute_error', 'validation_mean_absolute_error',
            'validation_sigmoid_cross_entropy', 'sigmoid_cross_entropy']

    # 'GPUUtil_0', 'GPUUtil_1', 'GPUUtil_3', 'GPUUtil_2', 'learning_rate', 'wd_cost','QueueInput_queue_size',
    fig = final_df[cols[-2:]].plot()
    fig.show()
    return final_df



def save_heatmap(in_cube, heatmap, overlay,path):
    # Depth x B-scan x A-scan
    r = 20
    c = 10
    fig, axes = plt.subplots(nrows=r, ncols=c, figsize=(30, 30))
    axes = axes.flatten()

    print(in_cube.shape, heatmap.shape, overlay.shape)
    in_cube = in_cube.transpose(2, 1, 0, 3)
    heatmap = heatmap.transpose(2, 1, 0, 3)
    overlay = overlay.transpose(2, 1, 0, 3)

    for i in range(in_cube.shape[1]):
        concat = np.concatenate(
            (in_cube[:, i, :, :].squeeze(), heatmap[:, i, :, :].squeeze(), overlay[:, i, :, :].squeeze()), axis=1)

        axes[i].imshow(concat)
        axes[i].axis('off')
    # cv2.imwrite('cam{}-{}.jpg'.format(i, t), concat)
    plt.savefig(path+'heatmaps.jpg', bbox_inches='tight')
    plt.show()

def normalize(array, min_value=0., max_value=1.):
    arr_min = np.min(array)
    arr_max = np.max(array)
    normalized = (array - arr_min) / (arr_max - arr_min + K.epsilon())
    return (max_value - min_value) * normalized + min_value


def resize_cube(cube, shape):
    """Return resized cube with the define shape"""

    zoom = [float(x) / y for x, y in zip(shape, cube.shape)]
    resized = sp.ndimage.zoom(cube, zoom)
    assert resized.shape == shape
    # print(cube.shape, resized.shape)
    return resized


def to_rgb(cube):
    cube = cube[..., np.newaxis]
    return np.concatenate(3 * (cube,), axis=-1)


def OverlayCamGAP_jet(cube, cam):
    shpe = (256, 200, 200)
    cube = resize_cube(cube, shpe)
    cube = normalize(cube, 0, 255).astype('uint8')

    cam = resize_cube(cam, shpe)
    cam_n = normalize(cam, 0, 255).astype('uint8')
    cam_jet = np.uint8(plt.cm.jet(cam_n)[..., :3] * 255)
    cam_jet[cam_n < 40, :] = 0

    alpha = 0.7
    cube_rgb = to_rgb(cube)

    overlay = (cube_rgb * alpha + cam_jet * (1. - alpha)).astype(cube_rgb.dtype)



    return cube_rgb, cam_jet, overlay


def get_garway_heathmap():
    arr = np.zeros((9, 9), np.int)  ### matches -OD eye and need to flip OS
    vf_garway_heath_map = {'superior': [1, 'yellow', 0], 'inferior': [2, 'green', 0], 'supperior_nasal': [3, 'red', 0],
                           'inferior_nasal': [4, 'blue', 0], 'central': [5, 'dark gray', 0],
                           'temporal': [6, 'light gray', 0],
                           'blind_spot': [7, 'white', 0]}
    arr[0, 3:7] = vf_garway_heath_map['superior'][0];
    arr[1, 2] = vf_garway_heath_map['superior'][0];
    arr[1, 6:8] = vf_garway_heath_map['superior'][0];
    arr[2, 7] = vf_garway_heath_map['superior'][0]
    arr[4, 0] = vf_garway_heath_map['inferior'][0];
    arr[5, 1] = vf_garway_heath_map['inferior'][0];
    arr[5, 7] = vf_garway_heath_map['inferior'][0];
    arr[6, 2:4] = vf_garway_heath_map['inferior'][0];
    arr[6, 6:8] = vf_garway_heath_map['inferior'][0];
    arr[7, 3:7] = vf_garway_heath_map['inferior'][0]
    arr[1, 3:6] = vf_garway_heath_map['supperior_nasal'][0];
    arr[2, 1:7] = vf_garway_heath_map['supperior_nasal'][0];
    arr[3, 0:4] = vf_garway_heath_map['supperior_nasal'][0]
    arr[4, 1:4] = vf_garway_heath_map['inferior_nasal'][0];
    arr[5, 2:7] = vf_garway_heath_map['inferior_nasal'][0];
    arr[6, 4:6] = vf_garway_heath_map['inferior_nasal'][0]
    arr[3, 4:7] = vf_garway_heath_map['central'][0];
    arr[4, 4:7] = vf_garway_heath_map['central'][0]
    arr[2:6, 8] = vf_garway_heath_map['temporal'][0]
    arr[3:5, 7] = vf_garway_heath_map['blind_spot'][0]

    # plt.imshow(arr, interpolation='nearest', cmap='gray', vmax=arr.max(), vmin=arr.min())
    # plt.show()
    return arr, vf_garway_heath_map


def Garway_Heathmap_Sectors(vft_gt, vft_pred, garway_map, heathmap_sectors):
    sectors = []
    for k, v in heathmap_sectors.items():
        # k is sector name , v[0] is in the index in the map , v[1] is the color, v[2] mean vft vals
        x_true = vft_gt[garway_map == v[0]].ravel()
        x_pred = vft_pred[garway_map == v[0]].ravel()
        gt_mean, pred_mean = np.mean(x_true), np.mean(x_pred)

        mae = mean_absolute_error(x_true, x_pred)
        r, pvalue = pearsonr(x_true, x_pred)
        sectors.append([k, mae, r, pvalue,gt_mean, pred_mean])

    return sectors

def perf_measures(ds, pred='', oct_type='',vft_type='THRESHOLD'):

    df = pd.read_csv('oct_onh_mac_vft_data_13119_folds.csv', index_col=0)
    cond = (df['TEST_PATTERN'] == 25) & (df['TEST_STRATEGY'] == 4)
    df = df[cond]
    garway_map_arr, heathmap_sectors_dict = get_garway_heathmap()
    for col in heathmap_sectors_dict.keys():
        mae_col_hm = 'mae_' + col
        pc_col_hm, pv_col_hm = 'r_' + col , 'pvalue_' + col
        gt_mean_col =  'gt_mean_' + col
        pred_mean_col =  'pred_mean_' + col

        df[mae_col_hm] = -1
        df[pc_col_hm] = -1
        df[pv_col_hm] = -1
        df[gt_mean_col] = -1
        df[pred_mean_col] = -1

    uid_col, sce_col, mse_col, mae_col, ssim_col = 'Name_' + oct_type, 'sce' , 'mse' , 'mae' , 'ssim'
    pc_col,pv_col = 'r' , 'p-value'


    df[mae_col] = -1
    df[mse_col] = -1
    df[sce_col] = -1
    df[ssim_col] = -1
    df[pc_col] = -1
    df[pv_col] = -1


    pred = SimpleDatasetPredictor(pred, ds)
    from constants import MD_range
    md_range = (np.abs(MD_range[0]) + (MD_range[1]))

    for outp_btch in pred.get_result():
        b_uid, b_vft_thresh, b_vft_pred, b_sce, b_mse, b_mae = outp_btch

        for o in zip(b_uid, b_vft_thresh, b_vft_pred, b_sce, b_mse, b_mae):
            uid, vft_thresh, vft_pred, sce, mse, mae = o

            vft_thresh = vft_thresh.astype(np.float32)

            if vft_type == 'THRESHOLD':
                vft_thresh = vft_thresh * 35
                vft_pred = vft_pred * 35
            elif vft_type == 'PATTERN':
                vft_thresh  = (vft_thresh * md_range) + MD_range[0]
                vft_pred = (vft_pred * md_range) + MD_range[0]

            df.loc[df[uid_col] == uid, mae_col] = mean_absolute_error(vft_thresh, vft_pred)#np.average(mae) 0-1
            df.loc[df[uid_col] == uid, mse_col] = mean_squared_error(vft_thresh, vft_pred)#np.average(mse) 0-1
            df.loc[df[uid_col] == uid, sce_col] = np.average(sce)
            df.loc[df[uid_col] == uid, ssim_col] = structural_similarity(vft_thresh, vft_pred)

            r,pval = pearsonr(vft_thresh.ravel(),vft_pred.ravel())

            df.loc[df[uid_col] == uid, pc_col] = r
            df.loc[df[uid_col] == uid, pv_col] = pval

            sectors_measures = Garway_Heathmap_Sectors(vft_thresh, vft_pred,garway_map_arr, heathmap_sectors_dict)
            for r in sectors_measures:
                df.loc[df[uid_col] == uid, 'mae_'+ r[0] ] = r[1]
                df.loc[df[uid_col] == uid, 'r_'+ r[0]] = r[2]
                df.loc[df[uid_col] == uid, 'pvalue_'+ r[0]] = r[3]

                df.loc[df[uid_col] == uid, 'gt_mean_' + r[0]] = r[4]
                df.loc[df[uid_col] == uid, 'pred_mean_' + r[0]] = r[5]
    return df.loc[df[mae_col] != -1]

def perf_measures_old(ds, pred='', oct_type=''):

    df = pd.read_csv('oct_onh_mac_vft_data_13119_folds.csv', index_col=0)
    cond = (df['TEST_PATTERN'] == 25) & (df['TEST_STRATEGY'] == 4)
    df = df[cond]
    garway_map_arr, heathmap_sectors_dict = get_garway_heathmap()
    for col in heathmap_sectors_dict.keys():
        mae_col_hm = 'mae_' + col + '_' + oct_type
        pc_col_hm, pv_col_hm = 'r_' + col + '_' + oct_type, 'pvalue_' + col + '_' + oct_type
        gt_mean_col =  'gt_mean_' + col + '_' + oct_type
        pred_mean_col =  'pred_mean_' + col + '_' + oct_type

        df[mae_col_hm] = -1
        df[pc_col_hm] = -1
        df[pv_col_hm] = -1
        df[gt_mean_col] = -1
        df[pred_mean_col] = -1

    uid_col, sce_col, mse_col, mae_col, ssim_col = 'Name_' + oct_type, 'sce_' + oct_type, 'mse_' + oct_type, 'mae_' + oct_type, 'ssim_' + oct_type
    pc_col,pv_col = 'r_' + oct_type, 'p-value_' + oct_type


    df[mae_col] = -1
    df[mse_col] = -1
    df[sce_col] = -1
    df[ssim_col] = -1
    df[pc_col] = -1
    df[pv_col] = -1


    pred = SimpleDatasetPredictor(pred, ds)

    for outp_btch in pred.get_result():

        b_uid, b_vft_thresh, b_vft_pred, b_sce, b_mse, b_mae = outp_btch

        for o in zip(b_uid, b_vft_thresh, b_vft_pred, b_sce, b_mse, b_mae):
            uid, vft_thresh, vft_pred, sce, mse, mae = o
            vft_thresh = vft_thresh.astype(np.float32)

            vft_thresh = vft_thresh * 35
            vft_pred = vft_pred * 35

            df.loc[df[uid_col] == uid, mae_col] = mean_absolute_error(vft_thresh, vft_pred)#np.average(mae) 0-1
            df.loc[df[uid_col] == uid, mse_col] = mean_squared_error(vft_thresh, vft_pred)#np.average(mse) 0-1
            df.loc[df[uid_col] == uid, sce_col] = np.average(sce)
            df.loc[df[uid_col] == uid, ssim_col] = structural_similarity(vft_thresh, vft_pred)

            r,pval = pearsonr(vft_thresh.ravel(),vft_pred.ravel())

            df.loc[df[uid_col] == uid, pc_col] = r
            df.loc[df[uid_col] == uid, pv_col] = pval

            sectors_measures = Garway_Heathmap_Sectors(vft_thresh, vft_pred,garway_map_arr, heathmap_sectors_dict)
            for r in sectors_measures:
                df.loc[df[uid_col] == uid, 'mae_'+ r[0] + '_' + oct_type] = r[1]
                df.loc[df[uid_col] == uid, 'r_'+ r[0] + '_' + oct_type] = r[2]
                df.loc[df[uid_col] == uid, 'pvalue_'+ r[0] + '_' + oct_type] = r[3]

                df.loc[df[uid_col] == uid, 'gt_mean_' + r[0] + '_' + oct_type] = r[4]
                df.loc[df[uid_col] == uid, 'pred_mean_' + r[0] + '_' + oct_type] = r[5]
    return df.loc[df[mae_col] != -1]

def get_perf_metrics(task='test', oct_type='onh', net_type='single', cols = None, print_measures=False,csv_path = None, csv_dir = None,fold =1):
    if csv_path is None:
        if csv_dir is None:
            print('You must pass a value for csv_dir or csv_path')
            return
        else:
            csv_path= csv_dir + 'perf_measures_oct-{}-f{}_input-{}.csv'.format(oct_type,fold,net_type)

    if cols is None:
        cols = ['mae_', 'mse_', 'sce_', 'ssim_']
    cols = [s + oct_type if s.endswith('_') else s for s in cols ]#,'mae_', 'mse_']]
    #cols[-1] = cols[-1]+'1'
    #cols[-2] = cols[-2] + '1'
    df_perf = pd.read_csv(csv_path)

    if fold == 0: # initial experiment
        if task != 'train':
            df_perf = df_perf[df_perf['Fold_1'] == task].iloc[:500, :]
        else:
            df_perf = df_perf[df_perf['Fold_1'] == task].iloc[:2000, :]
    else:
        fold_col = 'Fold_'+ str(fold)
        df_perf = df_perf[df_perf[fold_col] == task]

    from pprint import pprint
    if print_measures:
        pprint(df_perf[cols].describe().transpose().round(3))
    df = df_perf[cols]
    if cols is None:
        df.columns = [s.split('_')[0].upper() for s in cols]
    return df


def plot_bar_metrics(dic_df, by='task'):
    df_res = pd.DataFrame()
    pd.options.plotting.backend = "matplotlib"
    if isinstance(dic_df, dict):

        for task, df_perf in dic_df.items():
            df_perf['task'] = task
            df_res = df_res.append(df_perf)
    else:
        df_res = dic_df.copy()

    if by == 'task':
        df = df_res.groupby('task').mean().reset_index()
        df_err = df_res.groupby('task').std().reset_index()
    else:
        df = df_res.groupby('task').mean().transpose().reset_index()
        df_err = df_res.groupby('task').std().transpose().reset_index()

    # from pprint import pprint
    # pprint(df)
    ax = df.plot.bar(yerr=df_err, align='center', alpha=0.7, ecolor='black', capsize=6, rot=0, figsize=(15, 10))
    if by == 'task':
        ax.set_xticklabels(df['task'].values)
    else:
        ax.set_xticklabels(df['index'].values)
    ax.yaxis.grid(True)
    ax.legend()
    plt.show()
    ## Save the figure and show
    # plt.tight_layout()
    # plt.savefig('bar_plot_with_error_bars.png')
    # plt.show()
def plot_box_metrics_df(df, by_col = 'Fold_Num',cols = None):
    pd.options.plotting.backend = "plotly"
    fig = df.boxplot(column = cols, by= by_col)
    fig.show()
    return
def plot_box_metrics(oct_type='onh', net_type='single',cols=None, csv_path=None,csv_dir = None,fold=0):
    pd.options.plotting.backend = "plotly"
    #pd.options.plotting.backend = "matplotlib"
    dic_df = {}
    for task in ['Train','Val', 'Test']:
        print('#'*100)
        print('Performance: {}'.format(task) )
        print('#'*100)
        dic_df[task] = get_perf_metrics(task = task.lower(), oct_type = oct_type,net_type=net_type, print_measures=True, cols = cols,csv_path=csv_path,csv_dir = csv_dir,fold=fold)

    for task, df_perf in dic_df.items():
        fig=df_perf.boxplot(title = 'Performance: {}'.format(task))
        fig.show()
    return dic_df

def vft_details(uid='',oct_type='onh',df = pd.read_csv('oct_onh_mac_vft_data_13119_folds.csv', index_col=0),cols = ['TEST_PATTERN', 'TEST_STRATEGY', 'STIMULUS_COLOR',
       'STIMULUS_SIZE', 'BACKGROUND_COLOR', 'EXAM_DURATION', 'FIXATION_TARGET',
       'FIXATION_MONITOR', 'BLIND_SPOT_X', 'BLIND_SPOT_Y',
       'BLIND_SPOT_STIMULUS_SIZE', 'FALSE_NEGATIVE_METHOD',
       'FALSE_NEGATIVE_PERCENT', 'FALSE_POSITIVE_METHOD',
       'FALSE_POSITIVE_PERCENT', 'TRIALS', 'ERRORS', 'FOVEAL_RESULT',
       'FOVEAL_THRESHOLD', 'CENTRAL_REF_LEVEL', 'THROWN_OUT_POINTS']):
    if oct_type =='onh':
        uid_col = 'Name_onh'
    else:
        uid_col = 'Name_mac'
    return df.loc[df[uid_col] == uid , cols].to_dict()


def add_diagnosis_to_subjects(tasks=['Train', 'Val', 'Test'], oct_type='onh', net_type='single',
                              cols=['mae_', 'mse_', 'sce_', 'ssim_', 'subject_id_', 'eye_', 'mae_onh1', 'mse_onh1'],csv_path=None,fold = 0):
    dic_df = {}

    for task in tasks:
        dic_df[task] = get_perf_metrics(task=task.lower(), oct_type=oct_type, net_type=net_type, cols=cols, csv_path=csv_path,fold= fold)

    df_perf = pd.DataFrame()
    for task, df in dic_df.items():
        df['task'] = task
        df_perf = df_perf.append(df)
    df_perf = df_perf.reset_index()

    df_dx = pd.read_excel('/Users/gyasmeen/Desktop/Results/longitudinal/raw/health_stats.xlsx')
    df_perf['initial_dx'] = ''
    df_perf['secondary_dx'] = ''
    df_perf['primary_dx'] = ''

    c1 = 'subject_id'
    c2 = 'eye'

    for i in range(len(df_perf)):
        cond1 = df_dx[c1] == df_perf.loc[i, c1 + '_' + oct_type]
        cond2 = df_dx[c2] == df_perf.loc[i, c2 + '_' + oct_type]

        cond = cond1 & cond2

        if sum(cond) != 1:
            print(sum(cond), df_dx.loc[cond, 'initial_dx'].values[0])

        df_perf.loc[i, 'initial_dx'] = df_dx.loc[cond, 'initial_dx'].values[0]
        df_perf.loc[i, 'primary_dx'] = df_dx.loc[cond, 'primary_dx'].values[0]
        df_perf.loc[i, 'secondary_dx'] = df_dx.loc[cond, 'secondary_dx'].values[0]
    return df_perf


def plot_glaucoma_analysis(df_perf, analysis_col='primary_dx', by='metric', oct_type='onh',tasks=['Train', 'Val', 'Test']):
    '''
    cols = ['index', 'subject_id_' + oct_type, 'eye_' + oct_type, 'ss_' + oct_type, 'FALSE_POSITIVE_PERCENT','FALSE_NEGATIVE_PERCENT','ERRORS']
    if analysis_col in cols:
        cols.remove(analysis_col)
    df_perf = df_perf.drop(columns=cols)
    '''
    pd.options.plotting.backend = "matplotlib"

    for task in tasks:

        df_res = df_perf.loc[df_perf['FOLD_X'] == task]
        groupby = [analysis_col]
        if by == 'task':
            df = df_res.groupby(groupby).mean().reset_index()
            df_err = df_res.groupby(groupby).std().reset_index()
        else:
            df = df_res.groupby(groupby).mean().transpose().reset_index()
            df_err = df_res.groupby(groupby).std().transpose().reset_index()
        ax = df.plot.bar(yerr=df_err, align='center', alpha=0.7, ecolor='black', capsize=4, rot=0, figsize=(15, 10))
        if by != 'task':
            ax.set_xticklabels(df['index'].values)

        ax.yaxis.grid(True)
        ax.legend()
        ax.set_title(
            'VFT Prediction Model Performance Analysis: Healthy VS Glaucoma VS Suspects using {} diagnosis for {} Set of {} samples'.format(
                analysis_col.split('_')[0].upper(), task, len(df_res)))
        plt.show()


def plot_scatter_FP_FN_analysis(df_perf, tasks=['Train', 'Val', 'Test'], analysis_col='FALSE_POSITIVE_PERCENT',
                                 oct_type='onh', metric='ssim_'):
    pd.options.plotting.backend = "plotly"
    if tasks is None:
        df_res = df_perf.copy()
        metric = metric + oct_type if metric.endswith('_') else metric
        fig = df_res.plot.scatter(x=analysis_col, y= metric,
                                  title='VFT Prediction Model Performance Analysis: {} for the whole data Set of {} samples'.format(
                                      analysis_col, len(df_res)))
        fig.show()
    else:

        for task in tasks:
            df_res = df_perf.loc[df_perf['FOLD_X'] == task]
            fig = df_res.plot.scatter(x=analysis_col, y=metric + '_' + oct_type,
                                      title='VFT Prediction Model Performance Analysis: {} for {} Set of {} samples'.format(
                                          analysis_col, task, len(df_res)))
            fig.show()