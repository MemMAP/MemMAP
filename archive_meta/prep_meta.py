import functions as f
import pickle
from sklearn.model_selection import train_test_split
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, Sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.utils import plot_model
from keras.models import load_model, save_model
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
import keras.backend as K
from sys import getsizeof
import statsmodels.api as sm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from collections import Counter
import numpy as np
from numpy import argmax
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.rcParams['text.usetex'] = False
# matplotlib.use('Agg')
import time
import sys
from utils import *
import inspect
import collections
import glob
import pandas as pd
import functions as f

'''convert_to_dt'''
file_list = sorted(glob.glob("../data/*.out"))
dest_path="./data_dt/"
for i in range(len(file_list)):
    f.convert_to_dt(file_list[i], dest_path)
    print(file_list[i], "converted to delta sequence")


PROJECT_ROOT_DIRECTORY = "./" 
tok_save_path=PROJECT_ROOT_DIRECTORY+"pk_file/"
np_save_path=PROJECT_ROOT_DIRECTORY+"np_file/"
look_back=3


TRACE_FILE_NAMES = [
        'blackscholes',
        'bodytrack',
        'canneal',
        'dedup',
        'facesim',
        'ferret',
        'fluidanimate',
        'freqmine',
        'raytrace',
        'streamcluster',
        'swaptions',
        'vips',
        'x264'
    ]

for file in TRACE_FILE_NAMES:
    source_path_list = sorted(glob.glob("./data_dt/"+file+"*.csv"))
    dataset_conc=f.Concatenate_files(source_path_list,int(sys.argv[1]))
    print("concate done")  
    save_file_name=file
    f.Tokenize_and_Binarize(dataset_conc,save_file_name,tok_save_path,np_save_path,-1,look_back)#total_data_len, lookback
    