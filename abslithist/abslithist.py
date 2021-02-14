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
FIELD_DIR=os.path.join(PATH_DATA,'fields')
SOURCE_DIR=os.path.join(FIELD_DIR,'sources')
MODEL_DIR=os.path.join(PATH_DATA,'models')
PATH_FIELDS_JSON = os.path.join(FIELD_DIR,'data.fields_orig.json')
PATH_VECFIELDS_JSON = os.path.join(FIELD_DIR,'data.fields_vec.json')
PATH_FIELD2VEC_PKL = os.path.join(FIELD_DIR,'data.models.word_and_field_vecs.pkl')
PATH_NORMS = os.path.join(FIELD_DIR,'data.wordnorms_orig.csv')
VECNORMS_FN='data.wordnorms_vec.csv'
PATH_VECNORMS = os.path.join(FIELD_DIR,VECNORMS_FN)
if not os.path.exists(SOURCE_DIR): os.makedirs(SOURCE_DIR)
ZCUT = ZCUT_NORMS_ORIG
PATH_STOPWORDS=os.path.join(FIELD_DIR,'stopwords.txt')
PATH_NAMES=''