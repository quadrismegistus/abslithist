import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..'))
from abslithist import *

import gensim
from fastdist import fastdist




def get_model_paths(model_dir=PATH_MODELS,model_fn='model.txt.gz',vocab_fn='vocab.txt',period_len=None):
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



def load_model(path_model):
    path_vocab=path_model.replace('.txt.gz','.vocab.txt')
    if os.path.exists(path_vocab):
        model = gensim.models.KeyedVectors.load_word2vec_format(path_model,path_vocab)
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format(path_model)
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
        method=cdx['method']
        poswords=cdx['pos']
        negwords=cdx['neg']
        field2vec[f'{contrast}.{method}']=compute_vector(model,poswords,negwords)

    return field2vec


def compute_vec2vec_dists(x2vec,y2vec,xname='x',yname='y',distfunc='cosine'):
    func=getattr(fastdist,distfunc)
    ld=[]
    for x in x2vec:
        for y in y2vec:
            dx={
                xname:x,
                yname:y,
                'dist':func(
                    np.array(x2vec[x]),
                    np.array(y2vec[y])
                )
            }
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