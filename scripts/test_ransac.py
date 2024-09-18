import argparse
import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt


from fundamentalRANSAC import *


os.system('fundamentalRANSAC.py superglue -i1 ../new_pictures/torre_belem_6.jpg -i2 ../old_pictures/torre_de_belem_1930_1980.jpg \
          --iters 10000 --pathMatches ../superglue_output/torre_belem_6_torre_de_belem_1930_1980_matches.npz -t 12.0 -q')