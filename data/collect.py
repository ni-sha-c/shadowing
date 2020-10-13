# collect.py
import os
import argparse
from numpy import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, nargs='+')
parser.add_argument('-o', type=str)
args = parser.parse_args()

a = 0
for fname in args.i:
    a = a + load(fname)
for fname in args.i:
    os.remove(fname)
save(args.o, a)
