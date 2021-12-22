import numpy as np

import csv
import os


def read_csv_as2darray(path):
    arr = []
    with open(path, 'rU') as myFile:
        reader = csv.reader(myFile)
        for row in reader:
            arr.append(row)

    return np.asarray(arr)


def write_array2csv(arr, filename):
    with open(filename, 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for row in arr:
            wr.writerow(row)




def read_text(in_file):
    read_data=''
    with open(in_file) as f:
        read_data = f.read()
    return read_data


def write_text(out_file, text):
    with open(out_file,'w') as f:
        f.write(text)


