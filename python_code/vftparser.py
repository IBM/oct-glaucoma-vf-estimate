"""
Extracts information from Visual Field Test files in XML format.
"""

import xml.etree.ElementTree as et
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
from constants import *
import matplotlib
#matplotlib.use('TkAgg')

# matplotlib.use('Qt5Agg')
# matplotlib.rcParams['toolbar'] = 'None'

import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

from glob import glob
from datetime import datetime
BASEPATH = "/Users/gyasmeen/Desktop/Results/nyu_vft_xml/*.xml"

DISPLAY_NAME = 'PATIENT/STUDY/SERIES/DISPLAY_NAME'  # e.g. SS-24-2 Thr
VISIT_DATE = 'PATIENT/STUDY/VISIT_DATE'
SERIES_DATE_TIME = 'PATIENT/STUDY/SERIES/SERIES_DATE_TIME'
TEST_NODE = 'PATIENT/STUDY/SERIES/FIELD_EXAM/'
STATIC_TEST = TEST_NODE+'STATIC_TEST/'
STATPAC = STATIC_TEST + 'THRESHOLD_TEST/STATPAC/'
GLOBAL_INDICES = STATPAC + 'GLOBAL_INDICES/'

params = ['TEST_PATTERN', 'TEST_STRATEGY', 'STIMULUS_COLOR', 'STIMULUS_SIZE', 'BACKGROUND_COLOR', 'EXAM_DURATION',
          'FIXATION_TARGET' , 'FIXATION_MONITOR', 'BLIND_SPOT_X', 'BLIND_SPOT_Y', 'BLIND_SPOT_STIMULUS_SIZE', 'FALSE_NEGATIVE_METHOD',
          'FALSE_NEGATIVE_PERCENT' , 'FALSE_POSITIVE_METHOD', 'FALSE_POSITIVE_PERCENT', 'FIXATION_CHECK/TRIALS', 'FIXATION_CHECK/ERRORS',
          'FOVEAL_RESULT'  , 'FOVEAL_THRESHOLD', 'CENTRAL_REF_LEVEL', 'THROWN_OUT_POINTS', 'MINIMUM_STIMULUS', 'FIELD_SIZE','LANGUAGE',
          'THRESHOLD_TEST/SF_STATUS', 'THRESHOLD_TEST/NUM_THRESHOLD_POINTS']


class VFT(object):
    def __init__(self, fname, mat, gi):
        self.fname = fname
        self.mat = mat  # VFT as OS oriented matrix 8x9
        self.gi = gi  # global index, e.g. VFI
        self.dt = fname2dt(fname)  # visit time
        self.x = None  # embedding x-pos
        self.y = None  # embedding y-pos


def fname2dt(fname):
    """001797-2001-07-17-15-03-30-OD.xml -> 200107171503 """
    elems = [int(e) for e in fname.split('-')[1:7]]
    return datetime(*elems)


def extract_pattern_deviation(root):
    """Returns pattern deviation as sorted list of form [(x,y,value), ...]"""
    xpath = STATPAC + 'PATTERN_DEVIATION_VALUE_LIST/PATTERN_DEV_XY_LOCATION'
    vs = [[int(e.text) for e in elem] for elem in root.findall(xpath)]
    vs.sort()
    return vs


def extract_total_deviation(root):
    """Returns total deviation as sorted list of form [(x,y,value), ...]"""
    xpath = STATPAC + 'TOTAL_DEVIATION_VALUE_LIST/TOTAL_DEV_XY_LOCATION'
    vs = [[int(e.text) for e in elem] for elem in root.findall(xpath)]
    vs.sort()
    return vs


def extract_thresholds(root):
    """Returns test thresholds as sorted list of form [(x,y,value), ...]"""
    xpath = STATIC_TEST + 'THRESHOLD_TEST/THRESHOLD_SITE_LIST/THRESHOLD_XY_LOCATION'
    vs = [[int(e.text) for e in elem] for elem in root.findall(xpath)]
    #vs = [(x, y, v) for x, y, r, v in vs]  # filter out RESULT_1
    vs = [(row[0], row[1], row[3]) for row in vs]  # filter out RESULT_1
    vs.sort()
    return vs


def extract_vft_values(root, kind='THRESHOLD'):
    """VFT values: PATTERN, TOTAL, THRESHOLD"""
    if kind == 'PATTERN':
        return extract_pattern_deviation(root)
    if kind == 'TOTAL':
        return extract_total_deviation(root)
    if kind == 'THRESHOLD':
        return extract_thresholds(root)
    raise ValueError('Unknown VFT value kind: ' + kind)


def extract_global_index(root, kind='MD'):
    """Global VFT indices: MD, VFI, PSD"""
    xpath = GLOBAL_INDICES + kind
    elems = root.findall(xpath)
    if not elems: return None
    gi = float(elems[0].text)
    return gi

def extract_test_params(root):
    """VFT parameters, e.g. TEST_PATTERN, TEST_STRATEGY, ..."""
    res = {}
    '''
    xpath = STATIC_TEST + '*'
    elems = root.findall(xpath) + root.findall(xpath+'/FIXATION_CHECK*')
    #return {e.tag:int(e.text) for  e in elems if e.text.isdigit()}
    
    print(xpath)
    for e in elems:
        print(e.tag)
        if e.text.isdigit():
            res[e.tag] = int(e.text)
        elif len(e.text) > 1:
            #print(e.tag, e.text,type(e.text),'$'*100)
            res[e.tag] =e.text
        else:
            for ee in e:
                if ee.tag not in ['QUESTIONS_ASKED','SF']:

                    if ee.text.isdigit():
                        res[ee.tag] = int(ee.text)
                    elif len(ee.text) > 1:
                        res[ee.tag] = ee.text
    '''
    for p in params:
        xpath = STATIC_TEST + p
        el = root.findall(xpath)

        if not el:
            res[p.split('/')[-1]] =''
        elif el[0].text.isdigit():
            res[el[0].tag] = int(el[0].text)
        else:
            res[el[0].tag] = el[0].text


    for pth in [DISPLAY_NAME,VISIT_DATE,SERIES_DATE_TIME,TEST_NODE+'PUPIL_DIAMETER',TEST_NODE+'PUPIL_DIAMETER_AUTO',TEST_NODE+'EXAM_TIME']:
        e=root.find(pth)
        if e.text is None:
            res[e.tag] = e.text
        else:
            if e.text.isdigit():
                res[e.tag] = int(e.text)
            else:
                res[e.tag] = e.text
    '''
    vkind = ['THRESHOLD', 'TOTAL', 'PATTERN']

    for vk in vkind:
        vs = extract_vft_values(root, vk)
        mat = vf2matrix(vs)
        res[vk+'_MATRIX'] = [mat]
    '''

    return res

def extract_display_name(root):
    """VFT display name, e.g. SS-24-2 Thr"""
    elems = root.findall(STATIC_TEST)
    return elems[0].text if elems else None


def vf_dimensions(vs):
    """Min and max of VFT test point coordiantes"""
    xcoord = lambda v: v[0]
    ycoord = lambda v: v[1]
    xmin, xmax = min(vs, key=xcoord), max(vs, key=xcoord)
    ymin, ymax = min(vs, key=ycoord), max(vs, key=ycoord)
    return xmin[0], xmax[0], ymin[1], ymax[1]


def vf2matrix(vs,bg_val):
    """Convert VFT values to matrix"""
    c = 3 * 2
    vs = [(x // c, y // c, v) for x, y, v in vs]
    xmin, xmax, ymin, ymax = vf_dimensions(vs)
    mat = np.zeros((ymax - ymin + 1, xmax - xmin + 1)) + bg_val
    for x, y, v in vs:
        mat[y - ymin, x - xmin] = v
    return mat

def read_vft(filepath):
    """Read Visual Field Tests from XML file"""
    root = et.parse(filepath).getroot()
    return root


def is_25_4(root):
    """Return true if VFT is of type 25-4"""
    tp = extract_test_params(root)
    p, s = tp['TEST_PATTERN'], tp['TEST_STRATEGY']
    return (p, s) == (25, 4)

def normalize(mat,vrange):
    """Normalize to range [0...1]"""
    return (np.clip(mat, vrange[0], vrange[1]) - vrange[0])/(vrange[1]-vrange[0])

def normalize_vft(mat):
    """Normalize to range [0...1]"""
    return (np.clip(mat, -34, 1)-1)/-35.0

def read_vfts(n, gkind='MD', vkind='THRESHOLD', basepath=BASEPATH):
    """Read n Visual Field Tests from XML files"""
    vfts = []
    for fpath in sorted(glob(basepath)):#[:n]):
        fname = osp.basename(fpath).split('.')[0]

        try:
            root = read_vft(fpath)
        except:
            print("can't parse:" + fpath)
            continue
        gi = extract_global_index(root, gkind)
        # fpr = tp.get('FALSE_POSITIVE_PERCENT', 0)
        # fnr = tp.get('FALSE_NEGATIVE_PERCENT', 0)
        if not is_25_4(root) or gi is None:
            continue
        vs = extract_vft_values(root, vkind) # PATTERN TOTAL THRESHOLD
        mat = vf2matrix(vs)

        print(mat)
        ####mat = mat if fpath.endswith('-OS.xml') else np.fliplr(mat)
        assert mat.shape == (8, 9) # (8,9)
        vfts.append(VFT(fname, mat, gi))
    return vfts, fname, mat, gi

def view_vf(mat,tit):
    # fig = plt.figure(figsize=(1.5, 1.5), frameon=False)
    # #fig.canvas.window().statusBar().setVisible(False)
    # ax = plt.subplot()
    # ax.set_axis_off()
    # ax.imshow(mat, interpolation='nearest', cmap='gray', vmax=0, vmin=-30)


    plt.imshow(mat, interpolation='nearest', cmap='gray', vmax=mat.max(), vmin=mat.min())
    plt.title(tit)
    plt.show()

def xml_stats(basepath = BASEPATH):
    dfObj = None
    count  = 0
    for fpath in tqdm(sorted(glob(basepath))):  # [:n]):
        fname = osp.basename(fpath).split('.')[0]
        try:
            root = read_vft(fpath)
        except:
            print("can't parse:" + fpath)
            continue
        tp = extract_test_params(root)

        #p, s = tp['TEST_PATTERN'], tp['TEST_STRATEGY']
        #print('Testing Algorithm', tp, p, s)

        tp['Name']=fname
        '''
        gkind = ['VFI', 'MD', 'PSD']
        for gk in gkind:
            gi = extract_global_index(root, gk)
            tp[gk] = gi
        '''
        xpath = GLOBAL_INDICES
        types = ['MD','MD_PROBABILITY' ,'PSD','PSD_PROBABILITY','VFI','CPSD','CPSD_PROBABILITY','SF_PROBABILITY']

        for kind in types:
            e = root.findall(xpath+kind)

            if not e:
                tp[kind] = ''
            else:
                tp[e[0].tag] = float(e[0].text)
        xpath = STATPAC +'GHT'
        e = root.findall(xpath)

        tp[e[0].tag] = float(e[0].text)

        if dfObj is None:
            dfObj = pd.DataFrame(tp,index=[0])

        else:
            dfObj = dfObj.append(tp, ignore_index=True)

        count +=1
        if count == 200:
            dfObj.to_csv('xml_stats.csv', header=True, index=False)
            dfObj = None
        elif count % 200 == 0:
            dfObj.to_csv('xml_stats.csv', mode='a', header=False, index=False)
            dfObj = None
    if dfObj is not None:
        dfObj.to_csv('xml_stats.csv', mode='a', header=False, index=False)
def read_vft_ex(filepath):
    vkind = 'THRESHOLD'
    gkind='VFI'
    root = read_vft(filepath)
    gi = extract_global_index(root, gkind)
    vs = extract_vft_values(root, vkind)
    mat = vf2matrix(vs)

    if '-OS' in filepath:
        np.fliplr(mat)  # normalize laterality
    plt.imshow(mat, interpolation='nearest', cmap='gray', vmax=mat.max(), vmin=mat.min())
    plt.show()
    return mat #normalize_vft(mat) #+ [gi]
def get_garway_heathmap():
    arr = np.zeros((9, 9), np.int)  ### matches -OD eye and need to flip OS
    vf_garway_heath_map = {'superior': [1, 'yellow',0], 'inferior': [2, 'green',0], 'supperior_nasal': [3, 'red',0],
                           'inferior_nasal': [4, 'blue',0], 'central': [5, 'dark gray',0], 'temporal': [6, 'light gray',0],
                           'blind_spot': [7, 'white',0]}
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

    #plt.imshow(arr, interpolation='nearest', cmap='gray', vmax=arr.max(), vmin=arr.min())
    #plt.show()
    return arr, vf_garway_heath_map


def read_vft_heathmap(filepath, vkind='THRESHOLD',garway_map=None, heathmap_sectors=None):

    gkind = ['VFI', 'MD', 'PSD']
    if garway_map is None:
        garway_map, heathmap_sectors = get_garway_heathmap()

    root = et.parse(filepath).getroot()
    global_vft = [extract_global_index(root, gk) for gk in gkind]


    tp = extract_test_params(root)
    p, s = tp['TEST_PATTERN'], tp['TEST_STRATEGY']

    vs = extract_vft_values(root, vkind)

    mat = vf2matrix(vs,VFT_range[0] if vkind == 'THRESHOLD' else MD_range[0])
    if '-OS' in filepath:
        mat = np.fliplr(mat)  # normalize laterality
    empty_row = np.zeros((1, mat.shape[1]))
    mat = np.vstack([mat, empty_row])  # add empty row
    ###################################################################################
    if vkind =='THRESHOLD':
        mat = np.clip(mat, VFT_range[0], VFT_range[1])  # clip noisy values outside 0-35
    elif vkind == 'TOTAL':
        mat = np.clip(mat,MD_range[0],MD_range[1])
    sectors = []
    for k,v in heathmap_sectors.items():
        #k is sector name , v[0] is in the index in the map , v[1] is the color, v[2] mean vft vals
        val  = np.mean(mat[garway_map == v[0]])
        sectors.append(val)
        heathmap_sectors[k][2] = val

    if digitize_VFT:
        mat = np.digitize(mat, bins) # quantize into discrete bins
        mat = (mat * VFT_Mask).astype(np.float16)
        mat = to_categorical(mat, num_classes=len(bins))
    else:
        if vkind == 'THRESHOLD':
            mat = normalize(mat,VFT_range) # from 0-1
        elif vkind == 'PATTERN':
            mat = normalize(mat, MD_range)


    global_vft[0] = normalize(global_vft[0], VFI_range)
    global_vft[1] = normalize(global_vft[1], MD_range)
    global_vft[2] = normalize(global_vft[2], PSD_range)

    local_vft = mat


    return np.asarray(sectors), np.asarray(global_vft), local_vft

def explore_xml(filepath, vki='THRESHOLD'):
    root = et.parse(filepath).getroot()
    gkind = ['VFI', 'MD', 'PSD']
    for gk in gkind:
        gi = extract_global_index(root, gk)
        print(gk, gi)
    tp = extract_test_params(root)
    p, s = tp['TEST_PATTERN'], tp['TEST_STRATEGY']
    print('Testing Algorithm', tp, p, s)

    vkind = ['THRESHOLD', 'TOTAL', 'PATTERN']
    ind = 1
    for vk in vkind:
        vs = extract_vft_values(root, vk)

        mat = vf2matrix(vs,VFT_range[0] if vki == 'THRESHOLD' else MD_range[0])
        if '-OD' in filepath:
            np.fliplr(mat)  # normalize laterality
        print(len(vs), mat.shape)
        print(vs)
        plt.subplot(1, 3, ind)
        plt.imshow(mat, interpolation='nearest', cmap='gray', vmax=mat.max(), vmin=mat.min())
        plt.title(vk)
        ind += 1

    plt.show()



if __name__ == '__main__':
    basepath = '/Users/gyasmeen/Desktop/Results/nyu_vft_xml/'
    filename1 = '000001-2013-10-14-10-57-20-OD.xml'
    filename2 = '002155-2014-01-06-13-08-56-OS.xml'
    filepath = osp.join(basepath, filename1)
    explore_xml(filepath, vki='TOTAL')

    garway_map, sectors_info  = get_garway_heathmap()
    #plt.imshow(garway_map, interpolation='nearest', cmap='gray', vmax=garway_map.max(), vmin=garway_map.min())
    plt.show()


    i = 0
    for fname in ['002155-2014-01-06-13-08-56-OS.xml','001302-2010-02-04-14-26-48-OS.xml','002599-2012-09-24-13-29-26-OD.xml','001862-2008-06-03-20-40-50-OD.xml']:
        filepath = osp.join(basepath, fname)
        heathmap_sectors, global_vft, local_vft = read_vft_heathmap(filepath, vkind='TOTAL', garway_map=garway_map, heathmap_sectors=sectors_info)
        print(fname)
        print(local_vft)

        plt.subplot(1,4,i+1);plt.imshow(local_vft, interpolation='nearest', cmap='gray', vmax=local_vft.max(), vmin=local_vft.min())
        i+=1
    plt.show()

    #res = read_vft_ex(filepath)



    #xml_stats()
    # view_vf(extract_pattern_deviation(root))
    # extract_global_index(root)
    # extract_vft_values(root)


    # vfs, fname, mat, gi = read_vfts(2, gkind, vkind, BASEPATH)


