from scipy.stats import zscore
import pandas as pd

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






"""
Simple mofo'n parallelism with progress bar. Born of frustration with p_tqdm.
"""


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


def pmap_iter(func, objs, num_proc=4, use_threads=False, progress=True, desc=None):
	"""
	Yields results of func(obj) for each obj in objs
	Uses multiprocessing.Pool(num_proc) for parallelism.
	If use_threads, use ThreadPool instead of Pool.
	Results in any order.
	"""
	
	# imports
	import multiprocessing as mp
	from tqdm import tqdm
	
	# if parallel
	if num_proc>1:
		# create pool
		pool=mp.Pool(num_proc) if not use_threads else mp.ThreadPool(num_proc)

		# yield iter
		iterr = pool.imap_unordered(func, objs)
		for res in tqdm(iterr,total=len(objs),desc=desc) if progress else iterr:
			yield res
	else:
		# yield
		for obj in tqdm(objs) if progress else objs:
			yield func(obj)

def pmap(*x,**y):
	"""
	Non iterator version of pmap_iter
	"""
	# return as list
	return list(pmap_iter(*x,**y))