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
    for path in paths:#tqdm(paths,desc=f'Tokenizing and yielding skipgrams'):
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





def _do_save_skipgrams_corpus(obj):
    paths,ofn=obj
    save_skipgrams_from_paths(paths,ofn)

def save_skipgrams_corpus(C_lltk,period_len=50,min_year=None,max_year=None,num_proc=1):
    C=C_lltk
    oroot = f'data/models/{C.id}'
    df = C.metadata
    df['period']=df['year'].apply(lambda y: f'{y//period_len*period_len}-{y//period_len*period_len+period_len}')
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
    objs = [(paths,ofn) for paths,ofn in objs if not os.path.exists(ofn)]
        
    # exec
    pmap(
        _do_save_skipgrams_corpus,
        objs,
        num_proc=num_proc
    )
    