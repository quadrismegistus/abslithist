from abslithist import *
BY_SENTENCE=WORD2VEC_BY_SENTENCE


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
        skipgram+=[w for w in tokenize(sent.lower()) if w and w[0].isalpha()]
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
        num_proc=num_proc,
        desc='Tokenizing and yielding sentences'
    )
    

def gen_all_skipgrams(force=False,num_proc=1):
    gen_skipgrams_corpus('eebo_tcp',min_year=1500,max_year=1700,force=force,num_proc=num_proc)
    gen_skipgrams_corpus('ecco_tcp',min_year=1700,max_year=1800,force=force,num_proc=num_proc)
    gen_skipgrams_corpus('coha',min_year=1800,max_year=2000,force=force,num_proc=num_proc)


def restrict_w2v(w2v, restricted_word_set):
    """
    From https://stackoverflow.com/questions/48941648/how-to-remove-a-word-completely-from-a-word2vec-model-in-gensim
    """
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []
    w2v.init_sims(replace=True)

    for i in range(len(w2v.vocab)):
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
        vec_norm = w2v.vectors_norm[i]
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
            new_vectors_norm.append(vec_norm)

    w2v.vocab = new_vocab
    # w2v.vectors = new_vectors
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = new_index2entity
    w2v.index2word = new_index2entity
    w2v.vectors_norm = new_vectors_norm
    # w2v.init_sims()



def get_model_paths(model_dir=PATH_MODELS,model_fn='model.bin',vocab_fn='vocab.txt',period_len=None):
    """
    Get all models' paths
    """
    ld=[]
    for root,dirs,fns in os.walk(PATH_MODELS):
        if model_fn in fns:
            corpus,period,run=root.split('/')[-3:]
            if not 'run_' in run:
                corpus,period=root.split('/')[-2:]
                run=None
            dx={
                'corpus':corpus,
                'period_start':period.split('-')[0],
                'period_end':period.split('-')[-1],
                'path':os.path.join(root,model_fn),
                'path_vocab':os.path.join(root,vocab_fn)
            }
            if run is not None: dx['run']=run
            if period_len and int(dx['period_end'])-int(dx['period_start'])!=period_len:
                continue
            ld.append(dx)
    return ld

def filter_model(model,min_count=20):
    words_ok = {w for w in model.vocab if model.vocab[w].count>=min_count}
    restrict_w2v(model,words_ok)


def load_model(path_model,path_vocab=None,min_count=None):
    if path_model.endswith('.bin') and os.path.exists(path_model):
        model=gensim.models.Word2Vec.load(path_model,mmap='r')
    elif path_model.endswith('.txt.gz') and os.path.exists(path_model):
        if not path_vocab: path_vocab=os.path.join(os.path.dirname(path_model,'vocab.txt'))
        if os.path.exists(path_vocab):
            model = gensim.models.KeyedVectors.load_word2vec_format(path_model,path_vocab)
            if min_count: filter_model(model,min_count=min_count)
        else:
            model = gensim.models.KeyedVectors.load_word2vec_format(path_model)
    else:
        return
    return model






### Analysis
def get_centroid(model,words):
    vectors=[]
    for w in words:
        if w in model:
            vectors+=[model[w]]
    if not vectors: return None
    return np.mean(vectors,0)

def compute_vector(model,words_pos=[],words_neg=[]):
    centroid_pos=get_centroid(model,words_pos)
    if not words_neg: return centroid_pos
    centroid_neg=get_centroid(model,words_neg)
    if centroid_neg is not None:
        return centroid_pos - centroid_neg
    else:
        return centroid_pos


## compute
def compute_vecfields():
    #from abslithist.models.embeddings import get_model_paths,compute_word2field_dists
    from abslithist.words.fields import get_origfields

    # get paths
    pathld = get_model_paths()
    
    # load fields
    fields = get_origfields()

    # compute
    objs = [(pathd,fields) for pathd in pathld]
    all_data = pmap(compute_word2field_dists, objs, num_proc=4, desc='Computing word to field distances')






# This function computes all contast and vectors
def get_fieldvecs_in_model(model,fields={},contrasts=[]):
    """
    Compute field vectors in a model
    """
    ### LOAD
    field2vec={}

    for field in fields:
        field2vec[field]=compute_vector(model,fields[field])

    for cdx in contrasts:
        contrast=cdx['contrast']
        source=cdx['source']
        poswords=cdx['pos']
        negwords=cdx['neg']
        field2vec[f'{contrast}.{source}']=compute_vector(model,poswords,negwords)

    return field2vec


def compute_vec2vec_dists(x2vec,y2vec,xname='x',yname='y',distfunc='cosine'):
    func=getattr(fastdist,distfunc)
    ld=[]
    for x in x2vec:
        for y in y2vec:
            try:
                dx={
                    xname:x,
                    yname:y,
                    'dist':func(
                        np.array(x2vec[x]),
                        np.array(y2vec[y])
                    )
                }
            except Exception:
                continue
            ld.append(dx)
    df=pd.DataFrame(ld).pivot(xname,yname,'dist')
    return df



def compute_word2field_dists(obj,force=False):
    from abslithist.models.embeddings import compute_field_vectors
    # from scipy.spatial.distance import cosine
    from fastdist import fastdist
    import warnings
    warnings.filterwarnings('ignore')

    # input handle
    pathd,fields = obj
    path=pathd.get('path')
    if not path or not os.path.exists(path): return {}

    # load model
    model_dir = os.path.dirname(path)
    ofn_data = os.path.join(model_dir,'word2field_dists.csv')
    if not force and os.path.exists(ofn_data): return
    # load
    model = gensim.models.KeyedVectors.load_word2vec_format(path)

    # compute field vecs
    field2vec=compute_field_vectors(model,fields,incl_non_contrasts=False,incl_contrasts=True)

    ld=[]
    for word in model.vocab:
        for field in field2vec:
            dx={
                'word':word,
                'field':field,
                'cdist':fastdist.cosine(
                    np.array(model[word]),
                    np.array(field2vec[field])
                )
            }
            ld.append(dx)
    df=pd.DataFrame(ld).pivot('word','field','cdist')
    df['word_count']=[model.vocab[w].count for w in df.index]
    df['word_rank']=[model.vocab[w].index for w in df.index]
    # save
    df.sort_values('word_rank').to_csv(ofn_data)





#######
# Generating models
#######


# STONE

def _do_gen_model(obj):
    ifn,ofnfn,ofnfn_bin,ofnfn_vocab,attrs=obj
    # Load skips
    # skips = gensim.models.word2vec.LineSentence(ifn)
    skips = SkipgramsSampler(ifn,MODEL_NUM_SKIPS)
    # Gen model
	# model = gensim.models.Word2Vec(skips, workers=num_workers, sg=sg, min_count=min_count, size=num_dimensions, window=skipgram_size)
    model = gensim.models.Word2Vec(skips, **attrs)
    
    # Save model
    model.init_sims(replace=True)
    model.wv.save_word2vec_format(ofnfn, ofnfn_vocab)
    model.save(ofnfn_bin)


def gen_model(
        path_to_skipgram_file,
        skipgram_size=10,
        num_runs=1,
        num_skips_wanted=None,
        num_workers=8,
        min_count=MODEL_MIN_COUNT,
        num_dimensions=MODEL_NUM_DIM,
        sg=1,
        num_epochs=None,
        labels=[],
        num_proc=1
        ):
    import gensim,logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Load skipgrams
    # skips = gensim.models.word2vec.LineSentence(path_to_skipgram_file)

    print(f'Generating model ({num_runs} runs) of: {path_to_skipgram_file}')
    objs=[]
    for run in range(num_runs):
        # Output filename
        odir=os.path.join(os.path.dirname(path_to_skipgram_file),f'run_{str(run+1).zfill(2)}')
        if not os.path.exists(odir): os.makedirs(odir)
        ofnfn=os.path.join(odir,'model.txt.gz')
        ofnfn_bin=os.path.join(odir,'model.bin')
        if os.path.exists(ofnfn) or os.path.exists(ofnfn_bin): continue
        ofnfn_vocab=os.path.join(odir,'vocab.txt')

        obj = (
            path_to_skipgram_file,
            ofnfn,
            ofnfn_bin,
            ofnfn_vocab,
            dict(
                 workers=num_workers,
                 sg=sg,
                 min_count=min_count,
                 size=num_dimensions,
                 window=skipgram_size
            )
        )
        objs.append(obj)
    
    if objs:
        pmap(_do_gen_model, objs, num_proc=num_proc)

def gen_models(
        paths_to_skipgrams,
        **attrs
        ):
    for fn in paths_to_skipgrams:
        gen_model(fn,**attrs)

def gen_models_corpus(cname,period_len=MODEL_PERIOD_LEN,**attrs):
    skipgrams = sorted([
        d['path']
        for d in get_model_paths(model_fn='skipgrams.txt.gz',period_len=period_len)
        if d['corpus']==cname
    ])
    
    return gen_models(skipgrams,**attrs)

def gen_all_models(num_runs=10,num_proc=4):
    gen_models_corpus('eebo_tcp',num_runs=num_runs,num_proc=num_proc)
    gen_models_corpus('ecco_tcp',num_runs=num_runs,num_proc=num_proc)
    gen_models_corpus('coha',num_runs=num_runs,num_proc=num_proc)


####
# DISTS
####

def gen_top_words_across_models(words,ofn='data.top_words.json',num_proc=1,min_count=None,top_n=10):
    paths = get_model_paths()
    def _writegen():
        for gen in pmap_iter(
            _do_gen_top_words_across_models,
            [
                (words,pathd,min_count,top_n)
                for pathd in paths
            ],
            num_proc=num_proc,
            desc='Collecting top word associations across all models',
            progress=True
        ):
            for dx in gen:
                yield dx
    writegen_jsonl(os.path.join(DIST_DIR,ofn), _writegen)

def _do_gen_top_words_across_models(obj):
    words,pathd,min_count,top_n = obj
    modelmetad=dict((k,v) for k,v in pathd.items() if not k.startswith('path'))
    model = load_model(pathd['path'])
    vocab = model.vocab if hasattr(model,'vocab') else model.wv.vocab
    words = [w for w in words if w in vocab]
    if min_count: words = [w for w in words if vocab[w].count<min_count]
    res=[]
    # for word in tqdm(words,desc='Getting neighborhoods within model'):
    for word in words:
        top = model.wv.most_similar(word,topn=top_n)
        if top:
            dat = {'word':word, 'neighborhood':top, **modelmetad}
            res.append(dat)
            # yield dat
    return res


#### sem net?
