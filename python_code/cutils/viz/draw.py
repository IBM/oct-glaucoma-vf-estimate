import cv2
from ..cv.bwutils import find_contours
import  numpy as np
import random


def colors_gen(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r, g, b))
    return ret


def draw_rect(img, rect, col=(255, 0, 0)):
    """
    Draws rectangle in image
    :param im:
    :param rect: (top_left(x,y), (width, height))
    :return:
    """
    x1, y1, w, h = rect
    cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), col, 2)


def draw_rects(img, rects, colors=None):
    """
        Draws rectangles in image
    :param im:
    :param rect: rect array
    :return:
    """
    if (len(rects) > 0):
        if (colors is None):
            colors = colors_gen(len(rects))
        else:
            assert len(colors) == len(rects), 'number of colors and rectangles must match'

        for rect, col in zip(rects, colors):
            draw_rect(img, rect, col)


def draw_contours(im, mask, color=(0,255,0), thickness=3):
    cnts= find_contours(mask)
    if(len(im.shape)==2):
        im = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(im,cnts,0, color, thickness)
    return im


