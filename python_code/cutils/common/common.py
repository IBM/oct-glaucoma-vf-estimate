import numpy as np
import os


def normalize(X, target_range, source_range=None):
    if(source_range is None):
        xmin = np.min(X)
        xmax = np.max(X)
    else:
        xmin, xmax = source_range

    ratio = (target_range[1] - target_range[0]) * 1.0 / (xmax - xmin)
    # print 'ratio ', ratio
    X = target_range[0] + ratio * (X - xmin)
    return X


def normalize_range(X, source_range, target_range):
    xmin = source_range[0]
    xmax = source_range[1]
    ratio = (target_range[1] - target_range[0]) * 1.0 / (xmax - xmin)
    # print 'ratio ', ratio
    X = target_range[0] + ratio * (X - xmin)
    return X


def denormalize_range(Y, source_range, target_range):
    xmin = source_range[0]
    xmax = source_range[1]
    ratio = (target_range[1] - target_range[0]) * 1.0 / (xmax - xmin)
    # print 'ratio ', ratio
    Y = source_range[0] + (1/ratio)  * (Y - target_range[0])
    return Y



def mkdir(path, *args):
    """
    Gives a root path and and subpath, makes directory all the way from root to subpaths if they do not exist
    :param path: root path
    :param args:
    :return:
    """
    if(not os.path.exists(path)):
        os.mkdir((path))
    new_path = path
    for dir in args:
        new_path = os.path.join(new_path, dir)
        # print (new_path)
        if (not os.path.exists(new_path)):
            os.mkdir(new_path)
    return new_path




def save_keras_model(amodel, model_save_dir, prefix):
    print ('#Saving models')

    if (not os.path.exists(model_save_dir)):
        os.mkdir(model_save_dir)

    amodel.save(os.path.join(model_save_dir, prefix + '.hd5'))
    with open(os.path.join(model_save_dir, prefix + ".json"), "w") as text_file:
        text_file.write(amodel.to_json())

    print ("Done")




