#!/usr/bin/python3

import os
import sys
import argparse
from glob import glob
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='Autocrop png images.')
parser.add_argument('filenames', metavar='filename', type=str, nargs='+',
                    help='png image filenames to be cropped')

args = parser.parse_args()

for arg in args.filenames:
    for f in glob(arg):
        image=Image.open(f)
        image = image.convert('RGBA')

        image_data = np.asarray(image)
        not_white = (image_data.min(axis=2) < 255)
        if image_data.shape[2] > 3:
            not_white *= (image_data[:,:,3] > 0)
        non_empty_columns = np.where(not_white.max(axis=0))[0]
        non_empty_rows = np.where(not_white.max(axis=1))[0]
        cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

        image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]

        cropped = Image.fromarray(image_data_new)
        cropped.save(f)
