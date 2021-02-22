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


# def untokenize(tokens):
# 	psg=[]
# 	for tok in all_tokens:
# 		if tok in {"n't"} or not tok[0].isalpha():
# 			if psg:
# 				psg[-1]+=tok
# 				continue
# 		tokl=tok.lower()
# 		if tokl in dx['neg']: tok=f'<i><b>{tok}</b></i>'
# 		if tokl in dx['pos']: tok=f'<i><u>{tok}</u></i>'
# 		if tokl in dx['neither']: tok=f'<i>{tok}</i>'

# 		if psg and psg[-1] in {'(','['}):
# 			psg[-1]+=tok
# 		else:
# 			psg.append(tok)
# 	return ' '.join(psg)


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


def writegen_jsonl(fnfn,generator,args=[],kwargs={}):
	import jsonlines
	with jsonlines.open(fnfn,'w') as writer:
		for i,dx in enumerate(generator(*args,**kwargs)):
			writer.write(dx)
	print('>> saved:',fnfn)

def readgen_jsonl(fnfn):
	import jsonlines
	with jsonlines.open(fnfn) as reader:
		for dx in reader:
			yield dx

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

		# Close the pool?
		pool.close()
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



def do_pmap_group(obj):
	# unpack
	func,group_df,group_key,group_name = obj
	# load from cache?
	if type(group_df)==str:
		group_df=pd.read_pickle(group_df)
	# run func
	outdf=func(group_df)
	# annotate with groupnames on way out
	if type(group_name) not in {list,tuple}:group_name=[group_name]
	for x,y in zip(group_key,group_name):
		outdf[x]=y
	# return
	return outdf

def pmap_groups(func,df_grouped,use_cache=True,**attrs):
	import os,tempfile,pandas as pd
	from tqdm import tqdm

	# get index/groupby col name(s)
	group_key=df_grouped.grouper.names
	# if not using cache
	if not use_cache:
		objs=[
			(func,group_df,group_key,group_name)
			for group_name,group_df in df_grouped
		]
	else:
		objs=[]
		tmpdir=tempfile.mkdtemp()
		for i,(group_name,group_df) in enumerate(tqdm(list(df_grouped),desc='Preparing input')):
			tmp_path = os.path.join(tmpdir, str(i)+'.pkl')
			# print([i,group_name,tmp_path,group_df])
			group_df.to_pickle(tmp_path)
			objs+=[(func,tmp_path,group_key,group_name)]

	return pd.concat(
		pmap(
			do_pmap_group,
			objs,
			**attrs
		)
	)



### utils
def printm(x):
	from IPython.display import display,Markdown
	display(Markdown(x))


def writegen(fnfn,generator,header=None,args=[],kwargs={},find_all_keys=False,total=None,delimiter=','):
	from tqdm import tqdm
	import csv,gzip
	
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


	with (open(fnfn, 'w') if not fnfn.endswith('.gz') else gzip.open(fnfn,'wt')) as csvfile:
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



def draw_bokeh(G,
	title='Networkx Graph', 
	save_to=None,
	color_by=None,
	size_by=None,
	default_color='skyblue',
	default_size=15,
	min_size=5,
	max_size=30,
	show_labels=True
):
	import networkx as nx
	from bokeh.io import output_notebook, show, save
	from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges, LabelSet
	from bokeh.plotting import figure
	from bokeh.plotting import from_networkx
	from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
	from bokeh.transform import linear_cmap
	from networkx.algorithms import community
	from bokeh.plotting import from_networkx
	
	#Establish which categories will appear when hovering over each node
	HOVER_TOOLTIPS = [("ID", "@index")]#, ("Relations")]

	#Create a plot — set dimensions, toolbar, and title
	# possible tools are pan, xpan, ypan, xwheel_pan, ywheel_pan, wheel_zoom, xwheel_zoom, ywheel_zoom, zoom_in, xzoom_in, yzoom_in, zoom_out, xzoom_out, yzoom_out, click, tap, crosshair, box_select, xbox_select, ybox_select, poly_select, lasso_select, box_zoom, xbox_zoom, ybox_zoom, save, undo, redo, reset, help, box_edit, line_edit, point_draw, poly_draw, poly_edit or hover
	plot = figure(
		tooltips = HOVER_TOOLTIPS,
		tools="pan,wheel_zoom,save,reset,point_draw",
            active_scroll='wheel_zoom',
#             tools="",
		x_range=Range1d(-10.1, 10.1),
		y_range=Range1d(-10.1, 10.1),
		title=title
	)

	#Create a network graph object with spring layout
	# https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html

	#Set node size and color
	
	# size?
	size_opt = default_size
	if size_by is not None:
		size_opt = '_size'
		data_l = X = np.array([d.get(size_by,0) for n,d in G.nodes(data=True)])
		data_l_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
		data_scaled = [(min_size + (max_size * x)) for x in data_l_norm]
		for x,n in zip(data_scaled, G.nodes()):
			G.nodes[n]['_size']=x
			

	# get network
	network_graph = from_networkx(G, nx.spring_layout, scale=10, center=(0, 0))

	
	
	# render nodes
	network_graph.node_renderer.glyph = Circle(
		size=size_opt, 
		fill_color=color_by if color_by is not None else default_color
	)

	#Set edge opacity and width
	network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)

	#Add network graph to the plot
	plot.renderers.append(network_graph)

	#Add Labels
	if show_labels:
		x, y = zip(*network_graph.layout_provider.graph_layout.values())
		node_labels = list(G.nodes())
		source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
		labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='10px', background_fill_alpha=.7)
		plot.renderers.append(labels)

	show(plot)
	if save_to: save(plot, filename=save_to)

def get_numlines(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def tqdm_read_csv(fn,chunksize=10000,desc=None):
	from tqdm import tqdm
	import os,pandas as pd
	df_list=[]
	if not desc: desc=f'Reading CSV: {os.path.basename(fn)}'
	for df_chunk in tqdm(pd.read_csv(fn,chunksize=chunksize),total=get_numlines(fn),desc=desc):
		df_list.append(df_chunk)
	return pd.concat(df_list)