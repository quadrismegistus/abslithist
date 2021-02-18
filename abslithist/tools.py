from scipy.stats import zscore
import pandas as pd,re
import nltk

def zfy(series):
	series=pd.to_numeric(series,errors='coerce').dropna()
	series_z=pd.Series(zscore(series),index=series.index)
	return series_z

def download_tqdm(url, save_to):
	import requests
	from tqdm import tqdm

	r = requests.get(url, stream=True)
	total_size = int(r.headers.get('content-length', 0))

	with open(save_to, 'wb') as f:
		for chunk in tqdm(r.iter_content(32*1024), total=total_size, unit='B',unit_scale=True):
			if chunk:
				f.write(chunk)

	return save_to





def periodize(y):
	y=int(y)
	if bin_year_by==100:
		return f'C{(y//100) + 1}'
	else:
		return y//bin_year_by * bin_year_by



def display_source(code):
	import IPython
	def _jupyterlab_repr_html_(self):
		from pygments import highlight
		from pygments.formatters import HtmlFormatter

		fmt = HtmlFormatter()
		style = "<style>{}\n{}\n{}\n</style>".format(
			fmt.get_style_defs(".output_html"),
			fmt.get_style_defs(".jp-RenderedHTML"),
			'.jp-RenderedHTML span { font-size:0.8em }'
		)
		return style + highlight(self.data, self._get_lexer(), fmt)

	# Replace _repr_html_ with our own version that adds the 'jp-RenderedHTML' class
	# in addition to 'output_html'.
	IPython.display.Code._repr_html_ = _jupyterlab_repr_html_
	return IPython.display.Code(data=code, language="python3")

def source(x):
	from IPython.display import Code,display
	if type(x)!=str:
		import inspect
		x=inspect.getsource(x)
	display(display_source(x))


### TOKENIZER


def pmap_iter(func, objs, num_proc=4, use_threads=False, progress=True, desc=None):
	"""
	Yields results of func(obj) for each obj in objs
	Uses multiprocessing.Pool(num_proc) for parallelism.
	If use_threads, use ThreadPool instead of Pool.
	Results in any order.
	"""
	
	# imports
	from tqdm import tqdm
	
	# if parallel
	if desc: desc=f'{desc} [x{num_proc}]'
	if num_proc>1 and len(objs)>1:
		# create pool
		import multiprocessing as mp
		pool=mp.Pool(num_proc) if not use_threads else mp.pool.ThreadPool(num_proc)

		# yield iter
		iterr = pool.imap_unordered(func, objs)
		for res in tqdm(iterr,total=len(objs),desc=desc) if progress else iterr:
			yield res
	else:
		# yield
		for obj in (tqdm(objs,desc=desc) if progress else objs):
			yield func(obj)

def pmap(*x,**y):
	"""
	Non iterator version of pmap_iter
	"""
	# return as list
	return list(pmap_iter(*x,**y))


### utils
def printm(x):
	from IPython.display import display,Markdown
	display(Markdown(x))


def writegen(fnfn,generator,header=None,args=[],kwargs={},find_all_keys=False,total=None,delimiter=','):
	from tqdm import tqdm
	import csv
	
	if not header:
		iterator=generator(*args,**kwargs)
		if not find_all_keys:
			first=next(iterator)
			header=sorted(first.keys())
		else:
			print('>> finding keys:')
			keys=set()
			for dx in iterator:
				keys|=set(dx.keys())
			header=sorted(list(keys))
			print('>> found:',len(header),'keys')
	
	iterator=generator(*args,**kwargs)
	if total: iterator=tqdm(iterator,total=total)


	with open(fnfn, 'w') as csvfile:
		writer = csv.DictWriter(csvfile,fieldnames=header,extrasaction='ignore',delimiter=delimiter)
		writer.writeheader()
		for i,dx in enumerate(iterator):
			#for k,v in dx.items():
			#	dx[k] = str(v).replace('\r\n',' ').replace('\r',' ').replace('\n',' ').replace('\t',' ')
			writer.writerow(dx)
	print('>> saved:',fnfn)


def get_slices(l,num_slices=None,slice_length=None,runts=True,random=False):
	"""
	Returns a new list of n evenly-sized segments of the original list
	"""
	if random:
		import random
		random.shuffle(l)
	if not num_slices and not slice_length: return l
	if not slice_length: slice_length=int(len(l)/num_slices)
	newlist=[l[i:i+slice_length] for i in range(0, len(l), slice_length)]
	if runts: return newlist
	return [lx for lx in newlist if len(lx)==slice_length]
