import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..'))
from abslithist import *

BY_SENTENCE=True


### SKIPGRAMS
def yield_skipgrams_from_text(txt,skipgram_size=10,lowercase=True):
    skipgram=[]
    for token in tokenize(txt):
        skipgram+=[token]
        if len(skipgram)>=skipgram_size:
            x=skipgram[:skipgram_size]
            if x: yield x
            skipgram=[]

def yield_sentences_from_text(txt,skipgram_size=10,lowercase=True):
    skipgram=[]
    for sent in tokenize_sentences(txt):
        skipgram+=tokenize(sent)
        if len(skipgram)>=skipgram_size:
            yield skipgram
            skipgram=[]


def yield_skipgrams_from_paths(paths,by_sentence=BY_SENTENCE):
    for path in tqdm(paths,desc=f'Tokenizing and yielding skipgrams'):
        if not os.path.exists(path): continue
        with open(path) as f:
            txt=f.read()
            func = yield_sentences_from_text if by_sentence else yield_skipgrams_from_text
            for x in func(txt):
                yield x

def save_skipgrams_from_paths(paths,ofn,by_sentence=BY_SENTENCE):
    odir=os.path.dirname(ofn)
    if not os.path.exists(odir): os.makedirs(odir)
    with (open(ofn,'w') if not ofn.endswith('.gz') else gzip.open(ofn,'wt')) as of:
        for skip in yield_skipgrams_from_paths(paths,by_sentence=by_sentence):
            of.write(' '.join(skip)+'\n')


class SkipgramsSampler(object):
	def __init__(self, fn, num_skips_wanted):
		self.fn=fn
		self.num_skips_wanted=num_skips_wanted
		self.num_skips=self.get_num_lines()
		self.line_nums_wanted = set(random.sample(list(range(self.num_skips)), num_skips_wanted if num_skips_wanted<self.num_skips else self.num_skips))

	def get_num_lines(self):
		then=time.time()
		print('>> [SkipgramsSampler] counting lines in',self.fn)
		with gzip.open(self.fn,'rb') if self.fn.endswith('.gz') else open(self.fn) as f:
			for i,line in enumerate(f):
				pass
		num_lines=i+1
		now=time.time()
		print('>> [SkipgramsSampler] finished counting lines in',self.fn,'in',int(now-then),'seconds. # lines =',num_lines,'and num skips wanted =',self.num_skips_wanted)
		return num_lines

	def __iter__(self):
		i=0
		with gzip.open(self.fn,'rb') if self.fn.endswith('.gz') else open(self.fn) as f:
			for i,line in enumerate(f):
				line = line.decode('utf-8') if self.fn.endswith('.gz') else line
				if i in self.line_nums_wanted:
					yield line.strip().split()



def _do_save_skipgrams_corpus(obj):
    paths,ofn=obj
    save_skipgrams_from_paths(paths,ofn)

def gen_skipgrams_corpus(cname,period_len=MODEL_PERIOD_LEN,min_year=None,max_year=None,num_proc=1,force=False):
    import lltk
    C=lltk.load(cname)
    # C=C_lltk
    oroot = f'data/models/{C.id}'
    df = C.metadata
    # df['year']=df['year'].apply(lambda y: int(''.join(x for x in str(y) if x.isdigit())[:4]))
    df['period']=df['year'].apply(lambda y: f'{int(y)//period_len*period_len}-{int(y)//period_len*period_len+period_len}')
    if min_year: df=df[df.year>=min_year]
    if max_year: df=df[df.year<max_year]
        
    # prepare
    objs = [
        (
            [C.textd[idx].path_txt for idx in perioddf.id],
            os.path.join(oroot,period,'skipgrams.txt.gz')
        )
        for period,perioddf in sorted(df.groupby('period'))
    ]

    # remove already done
    objs = [(paths,ofn) for paths,ofn in objs if force or not os.path.exists(ofn)]
        
    # exec
    pmap(
        _do_save_skipgrams_corpus,
        objs,
        num_proc=num_proc
    )
    