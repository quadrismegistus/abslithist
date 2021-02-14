import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'.'))
from tools import *
from tqdm import tqdm
from collections import defaultdict
import requests,json,pandas as pd,numpy as np,pickle
import shutil,zipfile,random
import warnings
warnings.filterwarnings('ignore')

PATH_HERE = os.path.dirname(os.path.realpath(__file__))
PATH_ROOT = os.path.abspath(os.path.join(PATH_HERE,'..'))
PATH_DATA = os.path.abspath(os.path.join(PATH_ROOT,'data'))
PATH_MODELS = os.path.abspath(os.path.join(PATH_DATA,'models'))
ZCUT_NORMS_ORIG = 1.0
