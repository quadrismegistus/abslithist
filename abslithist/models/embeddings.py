import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..'))
from abslithist import *

import gensim
from scipy.spatial.distance import cosine




def get_model_paths(model_dir=PATH_MODELS,model_fn='model.txt.gz',vocab_fn='vocab.txt'):
    """
    Get all models' paths
    """
    ld=[]
    for root,dirs,fns in os.walk(PATH_MODELS):
        if model_fn in fns:
            corpus,period,run=root.split('/')[-3:]
            dx={
                'corpus':corpus,
                'period_start':period.split('-')[0],
                'period_end':period.split('-')[-1],
                'run':run,
                'path':os.path.join(root,model_fn),
                'path_vocab':os.path.join(root,vocab_fn)
            }
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
def compute_field_vectors(model, fields, incl_non_contrasts=False, incl_contrasts=True):
    """
    Compute field vectors in a model
    """
    
    ### LOAD
    # create dictionary from field -> vecs
    field2vec={}
    # First get non-contrasts
    if incl_non_contrasts:
        for field,words in fields.items():
            field2vec[field]=compute_vector(model,words)
    # Then get contrasts
    if incl_contrasts:
        contrast_methods = {
            tuple(fieldname.split('.')[:2]) for fieldname in fields
            if '-' in fieldname.split('.')[0]
        }
        for contrast,method in contrast_methods:
            contrast_pos,contrast_neg=contrast.split('-')
            key_pos = f'{contrast}.{method}.{contrast_pos}'
            key_neg = f'{contrast}.{method}.{contrast_neg}'
            key_neither = f'{contrast}.{method}.Neither'
            if key_pos in fields and key_neg in fields:
                field2vec[f'{contrast}.{method}']=compute_vector(model,fields[key_pos],fields[key_neg])
                #if key_neg in fields:
                #    field2vec[f'{contrast_pos}-Neither.{method}']=compute_vector(model,fields[key_pos],fields[key_neither])
                #    field2vec[f'{contrast_neg}-Neither.{method}']=compute_vector(model,fields[key_neg],fields[key_neither])

    return field2vec



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