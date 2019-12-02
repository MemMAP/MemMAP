from __future__ import division
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pandas.io import gbq
import logging
from datetime import datetime
from pylab import rcParams
from datetime import date, timedelta
import time
from sklearn.model_selection import train_test_split

from collections import defaultdict, Counter, OrderedDict
import sys
import os
import platform
import math
import pandas as pd
import shutil
import glob
import csv
import operator
import seaborn as sns
import pickle
import socket
import scipy as sc

from os import listdir
from os.path import isfile, join
from IPython.core.display import HTML, Image, display
import re
import requests
from shutil import copyfile
import seaborn;
import random
import string
from datetime import datetime

matplotlib.rcParams.update({'font.size': 12})

from dateutil.relativedelta import relativedelta
from IPython.display import set_matplotlib_formats


set_matplotlib_formats('retina')

from IPython.core.interactiveshell import InteractiveShell

# InteractiveShell.ast_node_interactivity = "all"

def save_obj(obj=None, name=None, directory=None, append=False):
    """
    Stores a an object to a pickle file. Since each dump open and closes the pickle file, append flag is needed if
    multiple appends need to be performed.

    :param obj:
    :param name:
    :param directory:
    :return:
    """
    if "/" not in directory:
        directory = directory + "/"
    with open(directory + name + '.pkl', '%sb' % ('a' if append else 'w')) as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name=None, directory=None, chunks=None):
    """
    Loads 1 or more pickle appends from a pickle file and returns a list of appends. If one append has happened, then
    this is equal to a simple picke loads. Otherwise, the resulting list will need some parsing to merge the entries.

    :param name:
    :param directory:
    :return:
    """

    objs = []
    if "/" not in directory:
        directory = directory + "/"
    f = open(directory + name + '.pkl', 'rb')
    index = 0
    while True:
        try:
            objs.append(pickle.load(f))
            print("Loading pickle...")
            index += 1
            if chunks and index >= chunks:
                print("Pickle Loaded.")
                break
        except EOFError:
            print("Pickle Loaded.")
            break
    f.close()
    if chunks == 1:
        return objs[0]
    else:
        return objs


def set_plot_size(width=None, height=None):
    if width and height:
        rcParams['figure.figsize'] = int(width), int(height)


def convert_timestamp(date=None):
    return date.replace("-", "")



def get_unix_timestamp(date_format=None, start_date=None, day_offset=None):
    if date_format is None:
        date_format = "%Y%m%d"
    if day_offset is not None:
        if start_date is not None:
            i = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=day_offset)
        else:
            i = datetime.utcnow() - timedelta(days=day_offset)
        return i.strftime(date_format)
    else:
        i = datetime.utcnow()
        return i.strftime(date_format)


def get_current_timestamp(format="%Y%m%d_%H%M%S"):
    return str(time.strftime(format))

def counter_hist(data=None):
    rcParams['figure.figsize'] = 45, 7
    labels, values = zip(*Counter(data).items())

    indexes = np.arange(len(labels))
    width = 0.7

    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.05, labels)
    plt.show()
    # print(Counter(data))



def setup_report(data_dir=None, readme_text=None, delete_old_results=True):
    if delete_old_results:
        try:
            shutil.rmtree(data_dir)
        except:
            pass

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if readme_text:
        with open("%s/%s" % (data_dir, "README.md"), 'w') as readme:
            readme.writelines(readme_text)



def set_plot_style_for_paper():
    params = {
        'axes.labelsize': 18,
        'font.size': 18,
        'legend.fontsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'text.usetex': True,
        #'figure.figsize': [4.5, 4.5],
        'figure.facecolor': 'w',
        'figure.edgecolor': 'w',
        'axes.facecolor': 'w',
        'axes.edgecolor': 'gray',
        'savefig.facecolor': 'w',
        'savefig.edgecolor': 'g',
        'savefig.pad_inches': 0.1,
        'savefig.transparent': True,
        'axes.titlepad': 20,
        'axes.titlesize': 19
    }
    rcParams.update(params)


