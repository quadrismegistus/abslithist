import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..'))
from abslithist import *
from abslithist.words import *


header_absconc=[
    'id',
    'slice',
    'source',
    'period',
    'num_abs',
    'num_conc',
    'num_neither',
    'num_total',
    'num_types',
    # 'num_words',
    'abs',
    'conc',
    'neither',
]



NORM_CONTRASTS=None

def get_norms_for_counting(sources={},periods={}):
    global NORM_CONTRASTS
    if NORM_CONTRASTS is None:
        NORM_CONTRASTS=[
            dx
            # for dx in get_origcontrasts()#get_allcontrasts(remove_stopwords=True)#get_allcontrasts()
            for dx in get_allcontrasts(remove_stopwords=True)
            if dx['source'] in SOURCES_FOR_COUNTING
        ]
    return [
        dx for dx in NORM_CONTRASTS
        if (not sources or dx['source'] in sources)
        and (not periods or dx['period'] in periods) 
    ]

def count_absconc_path(path,**attrs):
    if path.endswith('.gz') and os.path.exists(path[:-3]): path=path[:-3]
    with open(path,encoding='utf-8',errors='ignore') as f:
        ld=count_absconc(f.read(),**attrs)
        for dx in ld: dx['path']=path
        return ld

def count_absconc_path_psg(path,**attrs):
    with open(path,encoding='utf-8',errors='ignore') as f:
        ld=count_absconc(f.read(),sources={'Median'},periods={'median'},incl_psg=True,**attrs)
        for dx in ld: dx['path']=path
        return ld



def _count_absconc_window(dx,recog_tokens,all_tokens=[],incl_psg=False,psg_as_markdown=True,count_keys=['neg','pos','neither'],meta_keys=['contrast','source','period'],keyrename={'neg':'abs','pos':'conc'},vocab_len=5):
    only_words=[w for w in all_tokens if w and w[0].isalpha()]
    token_slice=recog_tokens
    # print(recog_tokens)
    tokenset=set(token_slice)
    countd=Counter(token_slice)
    cdx = {}
    # cdx['slice']=si+1
    cdx['num_words']=len(only_words) if only_words else len(recog_tokens)
    cdx['num_tokens']=len(token_slice)
    cdx['num_types']=len(tokenset)
    for key in meta_keys:
        cdx[key]=dx[key]
    
    total=0
    for key in count_keys:
        key2=keyrename.get(key,key)
        sharedwords=set(dx[key])&tokenset
        cdx['num_'+key2]=num=sum(countd[w] for w in sharedwords)
        total+=num
        if not incl_psg:
            cdx[key2]=', '.join(list(sorted(list(sharedwords),key=lambda w: -countd[w]))[:vocab_len])
    cdx['num_total']=total

    # include passage?
    if incl_psg:
        parens={'(',']'}
        psg=[]
        for tok in all_tokens:
            if tok in {"n't"} or (not tok[0].isalpha() and tok[0] not in parens):
                if psg:
                    psg[-1]+=tok
                    continue
            tokl=tok.lower()
            if tokl in dx['neg']: tok=f'<i><b>{tok}</b></i>'
            if tokl in dx['pos']: tok=f'<i><u>{tok}</u></i>'
            if tokl in dx['neither']: tok=f'<i>{tok}</i>'

            if psg and psg[-1] in parens:
                psg[-1]+=tok
            else:
                psg.append(tok)
        cdx['passage']=' '.join(psg)
    return cdx

def count_absconc(txt,window_len=COUNT_WINDOW_LEN,keep_last=True,periods={},sources={},
                incl_psg=False,psg_as_markdown=True,count_keys=['neg','pos','neither'],meta_keys=['contrast','source','period'],keyrename={'neg':'abs','pos':'conc'},vocab_len=5,modernize=MODERNIZE_SPELLING):
    # tokenize
    tokens = tokenize(txt,modernize=modernize)
    ld=[]
    for dx in get_norms_for_counting(sources=sources,periods=periods):
        # get all words known in this set
        allwords=set()
        for ck in count_keys: allwords|=dx[ck]

        all_tokens,recog_tokens=[],[]
        for i,tok in enumerate(tokens):
            # append
            tokl=tok.lower()
            all_tokens.append(tok)
            if tokl in allwords: recog_tokens.append(tokl)
            
            # ready?
            if len(recog_tokens)>=window_len:
                cdx=_count_absconc_window(dx,recog_tokens,all_tokens,incl_psg=incl_psg,psg_as_markdown=psg_as_markdown,count_keys=count_keys,meta_keys=meta_keys,keyrename=keyrename,vocab_len=vocab_len)
                if cdx:
                    cdx['slice']=len(ld)+1
                    cdx['tok_i']=i+1
                    ld.append(cdx)
                all_tokens,recog_tokens=[],[]
        # keep the last which is shorter?
        if keep_last:
            if all_tokens and recog_tokens:
                cdx=_count_absconc_window(dx,recog_tokens,all_tokens,incl_psg=incl_psg,psg_as_markdown=psg_as_markdown,count_keys=count_keys,meta_keys=meta_keys,keyrename=keyrename,vocab_len=vocab_len)
                if cdx:
                    cdx['slice']=len(ld)+1
                    cdx['tok_i']=i+2
                    ld.append(cdx)
        
    return ld



def count_absconc_corpus(cname,num_proc=1,save=True,ofn=None,eg_keys=['abs','conc','neither'],sample_n=10,incl_psg=False):

    
    # prepare
    import lltk
    C=lltk.load(cname)
    paths_txt = [t.path_txt.replace('.gz','') for t in C.texts() if os.path.exists(t.path_txt.replace('.gz',''))]
    path2id = dict((t.path_txt,t.id) for t in C.texts())
    # execute
    data = pmap_iter(
        count_absconc_path if not incl_psg else count_absconc_path_psg,
        paths_txt,
        num_proc=num_proc,
        desc=f'Counting abstract/concrete words in {cname}'
    )
    # reformat

    def _gen():
        for pathld in data:
            for dx in pathld:
                dx['id']=path2id.get(dx['path'])
                del dx['path']
                #for egk in eg_keys:
                #    dx[egk]=dx[egk][:sample_n]#random.sample(dx[egk],sample_n if len(dx[egk])>sample_n else dx[egk])
                # newld.append(dx)
                yield dx

    if not ofn: ofn=f'{COUNT_DIR}/data.absconc.{cname}{".psgs." if incl_psg else "."}v7.csv.gz'
    header = header_absconc if not incl_psg else [h for h in header_absconc if not h in eg_keys]+['passage']
    print('>> writing to:',ofn)
    writegen(ofn,_gen,header=header)
    # save as feather
    if not incl_psg: pd.read_csv(ofn).reset_index().to_feather(os.path.splitext(ofn)[0]+'.ft')
    
    
    # df=pd.DataFrame(newld)
    # # save?
    # if save:
    #     df.to_csv(ofn,index=False)
    # # return
    # return df
    
PSG_SOURCES={'Median'}
PSG_PERIODS={'median'}

def count_absconc_psg(txt,incl_psg=True,sources=PSG_SOURCES,periods=PSG_PERIODS,**attrs):
    df=pd.DataFrame(count_absconc(txt,incl_psg=incl_psg,sources=sources,periods=periods,window_len=100,**attrs))
    df['abs-conc']=df['num_abs']-df['num_conc']
    return df.sort_values('abs-conc')
    