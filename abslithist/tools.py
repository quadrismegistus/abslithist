from scipy.stats import zscore
import pandas as pd,re

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

SENTENCE_TOKENIZER=None

def get_sentence_tokenizer():
    global SENTENCE_TOKENIZER
    if SENTENCE_TOKENIZER is None:
        import stanza
        SENTENCE_TOKENIZER = stanza.Pipeline(lang='en', processors='tokenize')
    return SENTENCE_TOKENIZER

def tokenize_sentences_nlp(txt):
    nlp = get_sentence_tokenizer()
    return [
        #[token.text for token in sentence.tokens]
        sentence.text
		for sentence in nlp(txt).sentences
    ]

def tokenize_sentences(txt):
	import nltk
	return nltk.sent_tokenize(txt)


def tokenize(txt,lower=True):
	replacements = {
		0x2013: ' -- ',
		0x2014: ' -- ',
		0x201c: '"',
		0x201d: '"',
		0x2018: "'",
		0x2019: "'",
		0x2026: ' ... ',
		0xa0: ' '
	}
	for r in replacements:
		txt = txt.replace(chr(r), replacements[r])
	return tokenize_fast(txt)

def tokenize_fast(line,lower=True):
	import re
	from string import punctuation
	line=line.lower() if lower else line
	tokens = re.findall(
		r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
		# r'\w+',
		line
	)
	tokens = [w.strip(punctuation) for w in tokens]
	tokens = [w for w in tokens if w]
	return tokens

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
	if desc: desc=f'{desc} [x{num_proc}]'
	if num_proc>1 and len(objs)>1:
		# create pool
		pool=mp.Pool(num_proc) if not use_threads else mp.pool.ThreadPool(num_proc)

		# yield iter
		iterr = pool.imap_unordered(func, objs)
		for res in tqdm(iterr,total=len(objs),desc=desc) if progress else iterr:
			yield res
	else:
		# yield
		for obj in tqdm(objs,desc=desc) if progress else objs:
			yield func(obj)

def pmap(*x,**y):
	"""
	Non iterator version of pmap_iter
	"""
	# return as list
	return list(pmap_iter(*x,**y))