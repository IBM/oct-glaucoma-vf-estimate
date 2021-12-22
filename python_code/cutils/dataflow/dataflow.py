import os
import glob
import cv2
import numpy as np
from PIL import Image
import random
import abc
from scipy.ndimage import zoom



def resize(im, gt, size, is3d=False):
    """

    :param im:
    :param gt:
    :param size: (IMWxIMH) to resize (for 2d image)
    :return:
    """

    im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST)

    return (im, gt)




def resize3d(im, gt, size):
    """

    :param im:
    :param gt:
    :param size:  new size along each dimension
    :return:
    """

    assert len(im.shape) ==len(size)


    def f_rep(x,y):
        if(x is None):
            return y
        else:
            return x

    #dimension with  None in size will not change its shape
    newsize= [ f_rep(a,b) for a,b in zip(size,im.shape) ]


    ratios =  [(ns * 1.0) / s for s, ns in zip(im.shape, newsize)]

    im = zoom(im, ratios)
    gt = zoom(gt, ratios, order=0)


    return (im, gt)


def rotate_mult_channel_nearest(im, ang):
    nc = im.shape[2]
    imgs = []
    for i in range(nc):
        im_r = Image.fromarray(im[:, :, i]).rotate(ang, resample=Image.NEAREST)
        imgs.append(np.expand_dims(im_r, -1))
    return np.concatenate(imgs, axis=2)


def crop(im, gt, left, right, top, bot):
    h, w = im.shape[:2]

    im = im[top:h - bot, left:w - right]
    gt = gt[top:h - bot, left:w - right]
    return im, gt



def crop3d(im, gt, left, right, top, bot):
    h, w = im.shape[:2]

    im = im[top:h - bot, left:w - right]
    gt = gt[top:h - bot, left:w - right]
    return im, gt





def rotate(im, gt, ang):
    """

    :param im:
    :param gt:
    :param ang: in degrees

    :return:
    """

    # print 'rotate## ', im.shape, gt.shape
    if (ang is not 0):
        imr = Image.fromarray(im).rotate(ang, resample=Image.BILINEAR)
        if (len(gt.shape) <= 2):
            gtr = Image.fromarray(gt).rotate(ang, resample=Image.NEAREST)
        else:
            gtr = rotate_mult_channel_nearest(gt, ang)
    else:
        imr = im
        gtr = gt

    return (np.asarray(imr), np.asarray(gtr))


def rotate_cv2(im, gt, ang):
    """

    :param im:
    :param gt:
    :param ang: in degrees

    :return: rotated image and masks
    """

    if (ang is not 0):
        center = tuple(np.array(im.shape)[:2] / 2)
        rot_mat = cv2.getRotationMatrix2D(center, ang, 1.0)
        imr = cv2.warpAffine(im, rot_mat, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR)
        gtr = cv2.warpAffine(gt, rot_mat, (gt.shape[1], gt.shape[0]), flags=cv2.INTER_NEAREST)

    else:
        imr = im
        gtr = gt

    return (np.asarray(imr), np.asarray(gtr))


class DataFlow(object):
    abc.abstractmethod

    def get_data(self):
        """
        returns the generator of the data
        :return:
        """

    abc.abstractproperty

    def size(self):
        """
        returns the size of the data
        :return:
        """


class DataSummary(object):
    abc.abstractmethod

    def summarize(self):
        """
        Summarizes the data
        :return: the summary
        """


class BatchBuild(DataFlow):
    def __init__(self, ds, batch_size=None, loop_indefnite=False):
        """

        :param ds:
        :param batch_size: if None, returns all
        """
        self.ds = ds
        self.batch_size = batch_size
        self.loop_indefnite = loop_indefnite

        if (batch_size == None):
            assert not loop_indefnite, 'if batch_size is None, then loop_infinite should be False'

    def get_data(self):

        data = []
        label = []

        finished = False
        c = 0
        while (not finished):

            for d in self.ds.get_data():
                data.append(d[0])
                label.append(d[1])
                c = c + 1
                if (self.batch_size is not None and c >= self.batch_size):  # check if the next iteration wouldnt happen
                    data = np.asarray(data)
                    label = np.asarray(label)
                    yield [data, label]
                    data = []
                    label = []
                    c = 0

            if (not self.loop_indefnite):

                # return remaining
                if (len(data) > 0):
                    data = np.asarray(data)
                    label = np.asarray(label)
                    yield [data, label]
                finished = True




class BatchBuildGeneric(DataFlow):
    def __init__(self, ds, batch_size=None, loop_indefnite=False):
        """

        :param ds:
        :param batch_size: if None, returns all
        """
        self.ds = ds
        self.batch_size = batch_size
        self.loop_indefnite = loop_indefnite

        if (batch_size == None):
            assert not loop_indefnite, 'if batch_size is None, then loop_infinite should be False'

    def get_data(self):

        #data = None
        #label = []
        data=[]

        main_list_created=False
        finished = False
        c = 0
        while (not finished):

            for d in self.ds.get_data():
                if(not main_list_created):
                    temp = [data.append ([]) for e in d]
                    main_list_created = True
                for el, ml in zip(d, data):
                    ml.append(el)
                #data.append(d[0])
                #label.append(d[1])
                c = c + 1
                if (self.batch_size is not None and c >= self.batch_size):  # check if the next iteration wouldnt happen
                    toret=[]
                    for ml in data:
                        toret.append(np.asarray(ml))

                    yield toret
                    for ml in data:
                        ml=[]

                    c = 0

            if (not self.loop_indefnite):

                # return remaining
                if (len(data) > 0):
                    data = np.asarray(data)
                    label = np.asarray(label)
                    yield [data, label]
                finished = True

class Rotate(DataFlow):
    def __init__(self, ds, ang_mean, ang_std):
        self.ds = ds
        # self.size = size
        self.ang_mean = ang_mean
        self.ang_std = ang_std

    def size(self):
        return self.ds.size()

    def get_data(self):
        for d in self.ds.get_data():
            ang = np.random.normal(self.ang_mean, self.ang_std)

            im, gt = rotate_cv2(d[0], d[1], ang)
            d[0] =im
            d[1] = gt
            yield d #[im, gt]


class Crop(DataFlow):
    def __init__(self, ds, crop_params):
        """

        :param ds:
        :param crop_params: [ left, right, top, bot]
        """
        self.ds = ds
        self.crop_params = crop_params

    def size(self):
        return self.ds.size()

    def get_data(self):
        for d in self.ds.get_data():
            left, right, top, bot = self.crop_params
            im, gt = crop(d[0], d[1], left, right, top, bot)
            d[0] =im
            d[1] =gt
            yield d #[im, gt]



class Flip(DataFlow):
    def __init__(self, ds, prob=1):
        """

        :param ds:
        """
        self.ds = ds
        self.prob = prob

    def size(self):
        return self.ds.size()

    def get_data(self):
        for d in self.ds.get_data():
            if(random.random() <=self.prob):
                im,gt=d[0], d[1]
                im = cv2.flip(im, 1) #vertical flip
                gt = cv2.flip(gt, 1)  # vertical flip
                d[0] = im
                d[1] = gt


            yield d#[im, gt]



class Crop3D(DataFlow):
    def __init__(self, ds, crop_params):
        """

        :param ds:
        :param crop_params: [ left, right, top, bot]
        """
        self.ds = ds
        self.crop_params = crop_params

    def size(self):
        return self.ds.size()

    def get_data(self):
        for d in self.ds.get_data():

            im, gt = self.crop3d(d[0], d[1], self.crop_params)
            d[0] = im
            d[1] = gt
            yield d#[im, gt]

    def crop3d(self,im, gt, crop_params):
        z, h, w = im.shape
        z_up,z_down, left, right, top, bot = self.crop_params
        im = im[z_up:z - z_down,  top:h - bot, left:w - right]
        gt = gt[ z_up:z - z_down, top:h - bot, left:w - right]
        return im, gt



class Resize(DataFlow):
    def __init__(self, ds, newsize, is3d=False):
        """

        :param ds:
        :param size: imw ximh
        :param is3d  True if the image is a 3D image
        """
        self.ds = ds
        self.newsize = newsize
        self.is3d=is3d

    def size(self):
        return self.ds.size()

    def get_data(self):
        for d in self.ds.get_data():
            if(self.is3d):
                im, gt = resize3d(d[0], d[1], self.newsize)
            else:
                im, gt = resize(d[0], d[1], self.newsize)

            d[0] = im
            d[1] = gt
            yield d



class ResizeByComponent(DataFlow):
    def __init__(self, ds, newsize, apply_dim):
        """

        :param ds:
        :param size: imw ximh
        """
        self.ds = ds
        self.newsize = newsize
        self.apply_dim = apply_dim

    def size(self):
        return self.ds.size()

    def get_data(self):
        for d in self.ds.get_data():
            im = d[self.apply_dim]
            im = cv2.resize(im, self.newsize, interpolation=cv2.INTER_LINEAR)
            d[self.apply_dim] = im
            yield d


class DataMapper(DataFlow):
    def __init__(self, ds, map_funcs):
        self.ds = ds
        assert type(map_funcs) == list
        self.map_funcs = map_funcs

    def size(self):
        return self.ds.size()

    def get_data(self):

        for d in self.ds.get_data():

            dd = list(np.empty(len(d)))
            for c, f in enumerate(self.map_funcs):
                if f is not None:
                    dd[c] = f(d[c])
                else:
                    dd[c] = d[c]
            yield dd


class ExpandDims(DataFlow):
    def __init__(self, ds, dim0=None, dim1=None):
        self.ds = ds
        self.dim0 = dim0
        self.dim1 = dim1
        assert dim0 != None or dim1 != None, 'one of the dimension should be not none'

    def size(self):
        return self.ds.size()

    def get_data(self):

        for d in self.ds.get_data():
            im, gt=d[0], d[1]
            if (self.dim0 is not None):
                im = np.expand_dims(im, self.dim0)
            if (self.dim1 is not None):
                gt = np.expand_dims(gt, self.dim1)
            d[0] = im
            d[1] = gt
            yield d#[im, gt]


class Make2Class(DataFlow):
    def __init__(self, ds):
        """

        :param ds:
        :param size: imw ximh
        """
        self.ds = ds

    def size(self):
        return self.ds.size()

    def get_data(self):
        for d in self.ds.get_data():
            im, gt = d
            gt_neg = 1 - gt
            gt = np.concatenate([gt, gt_neg], axis=2)
            yield [im, gt]
