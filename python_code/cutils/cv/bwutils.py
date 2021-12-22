"""
Image processing for black and white images
"""

import cv2
import numpy as np


def remove_spirious_blobs(mask, area_threshold):
    """
    Removes the spiriors blobs from the mask whose area is less that area_thhreshold
    :param mask:
    :return: mask with  removed spirious blobs
    """

    contours = find_contours(mask)
    contours_new = []
    for c in contours:
        area = cv2.contourArea(c)
        if (area >= area_threshold):
            contours_new.append(c)

    mask_new = np.zeros(np.shape(mask))
    cv2.drawContours(mask_new, contours_new, -1, 255, -1)
    return mask_new


def find_contours(mask):
    """
    Computes contours in mask
    :param mask: contours in descending order of its area
    :return:
    """

    cv2_ver = cv2.__version__
    if cv2_ver[0] == '2' or cv2_ver[0] == '4':
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    return contours


def find_contours_with_max_area(mask):
    contours = find_contours(mask)
    maxA = 0
    cnt = contours[0]
    for c in contours:
        area = cv2.contourArea(c)
        if (area > maxA):
            cnt = c
            maxA = area

    return cnt


def filter_retain_largest_blob(mask):
    '''
    Removes all the blobs except the largest one from the mask
    :param im:
    :return new mask
    '''

    cnt = find_contours_with_max_area(mask)
    mask_new = np.zeros(np.shape(mask))
    cv2.drawContours(mask_new, [cnt], 0, 255, -1)

    return mask_new


def mask2boundingboxes(mask):
    """
    Compute the bounding box of each island in mask in descending order of its area
    :param mask:
    :return: list of rectangles [[x,y,w,h]]
    """

    contours = find_contours(mask)

    rects_sorted_area = []
    if (len(contours) > 0):
        maxA = 0
        cnt = contours[0]
        areas = []
        rects = []

        for c in contours:
            area = cv2.contourArea(c)
            areas.append(area)
            x, y, w, h = cv2.boundingRect(c)
            rect = [x, y, w, h]
            rects.append(rect)
        # print 'number of contours  ', len(rects)
        # sort in descending order of area
        rects_sorted_area = [(rects1, areas1) for (areas1, rects1) in
                             sorted(zip(areas, rects), key=lambda pair: pair[0], reverse=True)]
    # print rects_sorted

    rects_sorted = [rect for rect, area in rects_sorted_area]
    area_sorted = [area for rect, area in rects_sorted_area]

    return rects_sorted, area_sorted



def fill_bwpoly(bwim, arr):
    """

    :param bwim: single channel image
    :param arr: (N,2) array, each row is x,y point
    :return:
    """

    arr = arr.astype(np.int32)
    cv2.fillConvexPoly(bwim, arr,255)


def fill_hole(img, kernel = np.ones((5,5), np.uint8)):

    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    return img


def find_centroid(mask):
    """

    :param mask: black and white image
    :return: (x,y) center of the mask
    """
    # calculate moments of binary image
    M = cv2.moments(mask)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX,cY)