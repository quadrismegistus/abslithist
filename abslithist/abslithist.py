import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'.'))
from tools import *
from tqdm import tqdm
from collections import defaultdict,Counter
import requests,json,pandas as pd,numpy as np,pickle
import shutil,zipfile,random
import warnings,gzip,pickle
warnings.filterwarnings('ignore')
import plotnine as p9
p9.options.dpi=300
from scipy.stats import zscore
import math,time
import nltk



PATH_HERE = os.path.dirname(os.path.realpath(__file__))
PATH_ROOT = os.path.abspath(os.path.join(PATH_HERE,'..'))
PATH_DATA = os.path.abspath(os.path.join(PATH_ROOT,'data'))
PATH_MODELS = os.path.abspath(os.path.join(PATH_DATA,'models'))
PATH_LLTK_CORPORA = os.path.expanduser('~/lltk_data/corpora')
ZCUT_NORMS_ORIG = 1.0
FIELD_DIR=os.path.join(PATH_DATA,'fields')
SOURCE_DIR=os.path.join(FIELD_DIR,'sources')
MODEL_DIR=os.path.join(PATH_DATA,'models')
COUNT_DIR=os.path.join(PATH_DATA,'counts')
PSGS_DIR=os.path.join(PATH_DATA,'psgs')
DIST_DIR=os.path.join(PATH_DATA,'dists')
# PATH_FIELDS_JSON = os.path.join(FIELD_DIR,'data.fields_orig.json')
PATH_VECFIELDS_PKL= os.path.join(FIELD_DIR,'data.fields_vec.pkl')
PATH_VECFIELDS= os.path.join(FIELD_DIR,'data.fields_vec.csv.gz')
PATH_SPELLING_D = os.path.join(FIELD_DIR,'spelling_variants_from_morphadorner.txt')
PATH_NORMS = os.path.join(FIELD_DIR,'data.wordnorms_orig.csv')
VECNORMS_FN_PRE='data.wordnorms_vec'
VECNORMS_FN=f'{VECNORMS_FN_PRE}.csv'
PATH_VECNORMS = os.path.join(FIELD_DIR,VECNORMS_FN)
if not os.path.exists(SOURCE_DIR): os.makedirs(SOURCE_DIR)
ZCUT = ZCUT_NORMS_ORIG
PATH_STOPWORDS=os.path.join(FIELD_DIR,'stopwords.txt')
PATH_NAMES=os.path.join(FIELD_DIR,'capslocked.CanonFiction.txt')
MIN_COUNT_MODEL=25
MODEL_PERIOD_LEN=100
# MODEL_MIN_COUNT=25
MODEL_MIN_COUNT=10
MODEL_NUM_SKIPS=1000000
MODEL_NUM_DIM=100
# SOURCES_FOR_COUNTING = {'PAV-Conc','PAV-Imag','MRC-Conc','MRC-Imag','MT-Conc','LSN-Imag','Median'}
SOURCES_FOR_COUNTING = {'Median'}
SOURCES_FOR_PLOTTING = {'PAV-Conc','PAV-Imag','MRC-Conc','MRC-Imag','MT-Conc','LSN-Imag','Median'}
REMOVE_STOPWORDS_IN_WORDNORMS=True
MODERNIZE_SPELLING=False
COUNT_WINDOW_LEN=100
# COUNT_WINDOW_LEN=50