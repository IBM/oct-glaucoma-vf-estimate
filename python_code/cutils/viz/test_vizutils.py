from .vizutils import *
import numpy as np
import unittest


def test_get_heatmap_multiple():
    images = np.random.random((10, 320, 256))  # *255
    hms = get_heatmap_multiple(images)

    return hms.shape == (images.shape[0], images.shape[1], images.shape[2], 3)


class MyTest(unittest.TestCase):
    def test(self):
        assert (test_get_heatmap_multiple())


if (__name__ == '__main__'):
    print (test_get_heatmap_multiple())
