import os,sys,time,json,gzip,pickle,random
from collections import defaultdict,Counter
from tqdm import tqdm
import numpy as np,pandas as pd
from scipy.stats import zscore

ABSLITHIST_ROOT='/home/ryan/github/AbsLitHist'
VECDB_DIR=f'{ABSLITHIST_ROOT}/data/vecdb'
PATH_WORDNORM_ORIG=f'{ABSLITHIST_ROOT}/data/fields/data.wordnorms_orig.csv'
PATH_VECFIELDS_JSON=f'{ABSLITHIST_ROOT}/data/fields/data.vec.json.gz'
PATH_FIELDS_JSON=f'{ABSLITHIST_ROOT}/data/fields/data.fields.json'
PATH_FIELDS_PKL=f'{ABSLITHIST_ROOT}/data/fields/data.fields.orig_and_vec.pickle'
PATH_STOPWORDS=f'{ABSLITHIST_ROOT}/data/fields/stopwords.txt'
PATH_NAMES=f'{ABSLITHIST_ROOT}/data/fields/capslocked.CanonFiction.txt'
BAD_METHODS={'LSN-Act','Ness'}
GOOD_METHODS=['PAV-Conc','MRC-Conc','MT-Conc','PAV-Imag','MRC-Imag','LSN-Imag','LSN-Perc','LSN-Sens']
keytypes=['all','abs','conc','neither']
FIELDS=None


def get_vecfields():
	#with gzip.open(PATH_VECFIELDS_JSON) as f: d=json.load(f)
	with open(PATH_FIELDS_PKL,'rb') as f: d=pickle.load(f)
	return d
def get_origfields():
	with open(PATH_FIELDS_JSON) as f: d=json.load(f)
	return d

def get_fields(fieldprefix='Abs-Conc.',fieldsuffix='',remove_stopwords=True):
	global FIELDS
	if FIELDS is not None: return FIELDS
	## HEAVY LIFTING??
	now=time.time()
	stopwords=set() if not remove_stopwords else get_stopwords()

	# load orig
	origfields = get_origfields()
	# load vec'd
	vecfields = get_vecfields()
	# combine (bit adhoc: adding the neutrals)	
	fields = vecfields
	fields['Abs-Conc.PAV-Conc.Neither._orig'] = origfields['PAV.Conc.Neither']
	fields['Abs-Conc.PAV-Imag.Neither._orig'] = origfields['PAV.Imag.Neither']
	fields['Abs-Conc.MRC-Conc.Neither._orig'] = origfields['MRC.Conc.Neither']
	fields['Abs-Conc.MRC-Imag.Neither._orig'] = origfields['MRC.Imag.Neither']
	fields['Abs-Conc.MT.Neither._orig'] = origfields['MT.Neither']
	fields['Abs-Conc.LSN-Act.Neither._orig'] = origfields['LSN.Action.Neither']
	fields['Abs-Conc.LSN-Sens.Neither._orig'] = origfields['LSN.Sensorimotor.Neither']
	fields['Abs-Conc.LSN-Perc.Neither._orig'] = origfields['LSN.Perceptual.Neither']
	# filter
	fields2incl = [
		field for field in fields
		if (not fieldprefix or field.startswith(fieldprefix)) and (not fieldsuffix or field.endswith(fieldsuffix)) 
	]
	fields2excl = [field for field in fields if field not in set(fields2incl)]
	for field in fields2excl: del fields[field]
	for field in fields2incl:
		fields[field]=set(fields[field])-stopwords
	FIELDS=fields
	print('done.',time.time()-now)
	return fields



def get_stopwords(lower=True):
	paths = [
		path for path in 
		[PATH_STOPWORDS,PATH_NAMES]
		if os.path.exists(path)
	]
	stopwords=set()
	for path in paths:
		with open(path) as f:
			words=f.read().strip().split('\n')
			stopwords|=set((w.lower() if lower else w) for w in words)
	return stopwords





###
# loading models
###

def load_vecdf(vec,period=None):
    if not period: period='_median'
    path=f'{VECDB_DIR}/{vec}/{period}.csv'
    if os.path.exists(path):
        df=pd.read_csv(path).dropna()
        df['csim']=1-df['cdist']
        df['csim_z']=zscore(df.csim) #.apply(zscore,1)
        return df


def get_wordnorms_vec():
	"""
	@TODO: Loads vector norms (word csim's) for multiple vectors from vecdb
	"""
	pass



def get_wordnorms_orig():
	"""
	@TODO: Loads norms for words from original fields (QLD in SemantiFields ipynb)
	"""
	pass
















def get_absconcs(methods=[],periods=[],fields=None):
	if not fields: fields = get_fields()
	if not methods: methods=set([name.split('.')[1] for name in fields])-BAD_METHODS
	if not periods: periods=set([name.split('.')[3] for name in fields])
	# print(fields.keys())
	# print(methods)
	# print(periods)
	objs = [(method,period) for method in methods for period in periods]
	# print(objs)
	ld=[]
	for method,period in objs:
		# dx=get_absconc(method=method,period=period,fields=fields)
		# print(method,period,dx.keys())
		# if dx:
			# ld.append(dx)
		key_abs=f'Abs-Conc.{method}.Abs.{period}'
		key_conc=f'Abs-Conc.{method}.Conc.{period}'
		key_neither=f'Abs-Conc.{method}.Neither.{period}'
		nogo=False
		for key in [key_abs,key_conc]:#,key_neither]:
			if not key in fields:
				nogo=True
		if nogo: continue

		ld.append({
			'abs':fields[key_abs],
			'conc':fields[key_conc],
			'neither':fields.get(key_neither,set()),
			'all':fields[key_abs]|fields[key_conc]|fields.get(key_neither,set()),
			'method':method,
			'period':period,
			'prefix':'Abs-Conc'
		})
	return ld



# def get_absconc(method='ALL',period='_median',fields=None):
# 	if not fields: fields = get_fields(fieldprefix=f'Abs-Conc.{method}.',fieldsuffix=f'.{period}')
# 	key_abs=f'Abs-Conc.{method}.Abs.{period}'
# 	key_conc=f'Abs-Conc.{method}.Conc.{period}'
# 	key_neither=f'Abs-Conc.{method}.Neither.{period}'
# 	for key in [key_abs,key_conc]:#,key_neither]:
# 		if not key in fields:
# 			return {}

# 	return {
# 		'abs':fields[key_abs],
# 		'conc':fields[key_conc],
# 		'neither':fields.get(key_neither,set()),
# 		'all':fields[key_abs]|fields[key_conc]|fields.get(key_neither,set()),
# 		'method':method,
# 		'period':period,
# 		'prefix':'Abs-Conc'
# 	}

def tokenize(txt):
	import re
	#txt=txt.replace('\n','\\')
	txt=txt.replace('&hyphen;','-')
	#words=re.findall(r"[\w]+|[^\s\w]", txt)
	#words=[w if w!='\\' else '\n' for w in words]
	return [x for x in re.split('(\W+?)', txt) if x]

	# try:
	# 	if modernize_spelling:
	# 		words=[self.corpus.modernize_spelling(w) for w in words]
	# except AttributeError:
	# 	pass

	return words



def count_absconc(txt_or_tokens,methods=[],periods=[],summarize=False,fields=None):
	tokens = txt_or_tokens if type(txt_or_tokens)==list else tokenize(txt_or_tokens)
	tokens = [w.lower() for w in tokens if w.isalpha()]
	counts = Counter(tokens)
	absconcld = get_absconcs(methods=methods, periods=periods, fields=fields)
	stopwords = get_stopwords()
	num_content_words=sum(v for k,v in counts.items() if k not in stopwords)

	old=[]
	for idx in absconcld:
		odx={'method':idx['method'], 'period':idx['period'],'prefix':idx['prefix'], 'total_all':len(tokens), 'total':num_content_words}
		total_recog=0
		for key in ['all','abs','conc','neither']:
			odx['num_'+key]=vx=sum(v for k,v in counts.items() if k in idx[key])
			if key!='all': odx['perc_'+key]=odx['num_'+key]/odx['num_all']*100 if odx['num_all'] else np.inf
		odx['abs/conc']=odx['num_abs']/odx['num_conc'] if odx['num_conc'] else np.inf
		old.append(odx)
	
	# for key in ['abs','conc','neither']:
		# df[f'perc_{key}'] = df[f'num_{key}'].sum()
	return old if not summarize else count_absconc_summarize(old)				

def count_absconc_summarize(ld):
	df=pd.DataFrame(ld)
	odx={'total_all':df['total_all'].median(), 'total':df['total'].median()}
	for key in keytypes:
		odx[f'num_{key}']=df[f'num_{key}'].sum()
	for key in keytypes:
		if key != 'all':
			odx[f'perc_{key}']=odx[f'num_{key}']/odx['num_all']*100 if odx['num_all'] else np.inf
	odx['abs/conc*10']= odx['num_abs'] / odx['num_conc'] * 10
	odx['abs/conc']= odx['num_abs'] / odx['num_conc']
	return odx

def highlight(txt,incl_stat=True,**x):
	absconcd = get_absconc(**x)
	newtxtl=[]
	num_abs=0
	num_conc=0
	num_neither=0
	for w in tokenize(txt):
		wl=w.lower()
		if not w.isalpha() or not w.strip():
			w2=w
		elif wl in absconcd['conc']:
			num_conc+=1
			w2=f'<b>{w}</b>'
		elif wl in absconcd['abs']:
			num_abs+=1
			w2=f'<u><i>{w}</i></u>'
		elif wl in absconcd['neither']:
			num_neither+=1
			w2=w#f'<i>{w}</i>'
		else:
			w2=w#f'{w}'
		newtxtl+=[w2]
	
	# add stats?
	if incl_stat:
		ttl=num_abs+num_conc+num_neither
		perc_abs=int(round(num_abs/ttl*100))
		perc_conc=int(round(num_conc/ttl*100))
		ratio=round(num_abs/num_conc,1) if num_conc else np.inf
		newtxtl.append(f"""

({ttl} content words, of which {num_abs} abstract ({perc_abs}%) & {num_conc} concrete ({perc_conc}%); their ratio is {ratio})
""")
	
	psg=''.join(newtxtl)

	return psg




### utils
def printm(x):
    from IPython.display import display,Markdown
    display(Markdown(x))