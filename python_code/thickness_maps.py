import cv2
import numpy as np

###pip install cutil
from cutils.cv.bwutils import remove_spirious_blobs, fill_hole


def gen_thickness_map(segmap, layer_index, axial_resolution=255, exclude_disc=True, pvars=None, pvar_thresh=None):
    """

    :param segmap: segmentation map
    :param layer_index: index of layer of which thikcness map is to be computed
    :param axial_resolution: axisal resolution of the original oct image
    :param exclude_disc:
    :param pvars: (optional) variance associated with the prediction of segmap
    :param pvar_thresh: (optional) the voxels where pvars > pvar_thres will be not included while computed to compute the map
    :return: the thickness map of the given layer
    """

    layer_map = segmap == layer_index
    if(pvars is not None):
        assert pvar_thresh is not None, 'threshold should be provide'
        mask_ignore = pvars > pvar_thresh
        layer_map = layer_map * ~mask_ignore

    rnfl_tmap = np.sum(layer_map, axis=1)
    rnfl_tmap = (rnfl_tmap/rnfl_tmap.max()) * axial_resolution


    disc_mask = compute_disc(segmap)

    non_disc = 1 - disc_mask/255.0
    #disc_mask = cup_mask

    if (exclude_disc):
        rnfl_tmap = non_disc * rnfl_tmap

    return rnfl_tmap, disc_mask


def compute_disc(segmap):
    #disc_mask = (np.sum(segmap == 7, axis=1) <=4).astype(np.uint8)*255
    #disc_mask = remove_spirious_blobs(disc_mask , 50)
    #disc_mask = fill_hole(disc_mask)


    # not include gcl to compute mask as gcl can vanish sometime
    non_disc = (np.sum(segmap == 2, axis=1) > 0) * \
                (np.sum(segmap == 3, axis=1) > 0) * \
                (np.sum(segmap == 4, axis=1) > 0)
    disc_mask = (1 - non_disc.astype(np.uint8))*255
    disc_mask = remove_spirious_blobs(disc_mask , 50)
    disc_mask = fill_hole(disc_mask)

    return disc_mask.astype(np.uint8)





def gen_enface_projection(cube):

    #octma = np.ma.array(oct, mask=segmap != 7)
    #proj_max = np.max(octma, axis=1).astype(np.uint8)

    proj = np.max(cube, axis=1).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    proj = clahe.apply(proj)
    return proj.astype (np.uint8)

def gen_enface_projection_rpe(oct_cube, segmap=None):

    octma = np.ma.array(oct_cube, mask=segmap != 7)
    proj_mean = np.mean(octma, axis=1).astype(np.uint8)
    return proj_mean.astype (np.uint8)


import cv2
from glob import glob
import numpy as np
import os

def save_thicknesses(layer_index=[0, 1], layer_name=['RNFL', 'GCIPL'], load_dir='/Users/gyasmeen/Downloads/'):
    files_list = glob(load_dir + 'layerSeg/*.npy')

    for i in range(len(layer_index)):
        save_dir = load_dir + layer_name[i] + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for fpath in files_list:
            fname = fpath.split('/')[-1]
            segmap = np.load(fpath)
            tmap, disc_mask = gen_thickness_map(segmap, layer_index[i], axial_resolution=255, exclude_disc=True,
                                                pvars=None, pvar_thresh=None)

            cv2.imwrite(save_dir + fname.split('.')[0] + '_thickness.png', tmap)
            cv2.imwrite(save_dir + fname.split('.')[0] + '_disc.png', disc_mask)


if __name__ == "__main__":
    save_thicknesses(layer_index=[0, 1], layer_name=['RNFL', 'GCIPL'], load_dir='/Users/gyasmeen/Downloads/')

