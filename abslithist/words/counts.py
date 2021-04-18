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
<<<<<<< HEAD
    'abs',
    'conc',
    'neither',
    'passage'
=======
    # 'num_words',
    'abs',
    'conc',
    'neither',
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41
]



NORM_CONTRASTS=None

<<<<<<< HEAD
def get_norms_for_counting(sources=SOURCES_FOR_COUNTING,periods=PERIODS_FOR_COUNTING):
    global NORM_CONTRASTS
    if NORM_CONTRASTS is None:
        # print('>> loading data')
=======
def get_norms_for_counting(sources={},periods={}):
    global NORM_CONTRASTS
    if NORM_CONTRASTS is None:
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41
        NORM_CONTRASTS=[
            dx
            # for dx in get_origcontrasts()#get_allcontrasts(remove_stopwords=True)#get_allcontrasts()
            for dx in get_allcontrasts(remove_stopwords=True)
<<<<<<< HEAD
            # if dx['source'] in SOURCES_FOR_COUNTING
=======
            if dx['source'] in SOURCES_FOR_COUNTING
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41
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



<<<<<<< HEAD
def _count_absconc_window(
        dx,
        recog_tokens,
        all_tokens=[],
        incl_psg=False,
        incl_eg=True,
        psg_as_markdown=True,
        count_keys=['neg','pos','neither'],
        meta_keys=['contrast','source','period'],
        keyrename={'neg':'abs','pos':'conc'},
        num_eg=10,
        markdown_uses_html=False
        ):
=======
def _count_absconc_window(dx,recog_tokens,all_tokens=[],incl_psg=False,psg_as_markdown=True,count_keys=['neg','pos','neither'],meta_keys=['contrast','source','period'],keyrename={'neg':'abs','pos':'conc'},vocab_len=5):
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41
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
<<<<<<< HEAD
        if incl_eg:
            egs=[
                f'{w} ({countd[w]})' if countd[w]>1 else w
                for w in sorted(sharedwords,key=lambda w: (-countd[w],token_slice.index(w.lower())))
            ][:num_eg]
            cdx[key2]=', '.join(egs)
=======
        if not incl_psg:
            cdx[key2]=', '.join(list(sorted(list(sharedwords),key=lambda w: -countd[w]))[:vocab_len])
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41
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
<<<<<<< HEAD

            if markdown_uses_html:
                if tokl in dx['neg']: tok=f'<i><b>{tok}</b></i>'
                if tokl in dx['pos']: tok=f'<i><u>{tok}</u></i>'
                if tokl in dx['neither']: tok=f'<i>{tok}</i>'
            else:
                if tokl in dx['pos']: tok=f'```{tok}```'
                # if tokl in dx['neg']: tok=f'```{tok}```'
                # if tokl in dx['pos']: tok=f'***{tok}***'
                if tokl in dx['neg']: tok=f'***{tok}***'
                if tokl in dx['neither']: tok=f'*{tok}*'
=======
            if tokl in dx['neg']: tok=f'<i><b>{tok}</b></i>'
            if tokl in dx['pos']: tok=f'<i><u>{tok}</u></i>'
            if tokl in dx['neither']: tok=f'<i>{tok}</i>'
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41

            if psg and psg[-1] in parens:
                psg[-1]+=tok
            else:
                psg.append(tok)
<<<<<<< HEAD
        psg=' '.join(psg)
        psg=psg.replace('** **',' ').replace('``` ```',' ')
        cdx['passage']=psg
    return cdx

def count_absconc(
        txt,
        window_len=COUNT_WINDOW_LEN,
        keep_last=True,
        periods={},
        sources={},
        incl_psg=False,
        psg_as_markdown=True,
        count_keys=['neg','pos','neither'],
        meta_keys=['contrast','source','period'],
        keyrename={'neg':'abs','pos':'conc'},
        num_eg=10,
        modernize=MODERNIZE_SPELLING,
        progress=False,
        **attrs):
    # tokenize
    txt=txt.replace('\r\n','\n').replace('\r','\n').replace('\n',' \\\\ ').replace("`","'")
    tokens = tokenize(txt,modernize=modernize,lower=False)
    ld=[]
    iterr=get_norms_for_counting(sources=sources,periods=periods)
    if progress: iterr=tqdm(iterr)
    for dx in iterr:
        # print(dx['source'],dx['period'])
=======
        cdx['passage']=' '.join(psg)
    return cdx

def count_absconc(txt,window_len=COUNT_WINDOW_LEN,keep_last=True,periods={},sources={},
                incl_psg=False,psg_as_markdown=True,count_keys=['neg','pos','neither'],meta_keys=['contrast','source','period'],keyrename={'neg':'abs','pos':'conc'},vocab_len=5,modernize=MODERNIZE_SPELLING):
    # tokenize
    tokens = tokenize(txt,modernize=modernize)
    ld=[]
    for dx in get_norms_for_counting(sources=sources,periods=periods):
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41
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
<<<<<<< HEAD
                cdx=_count_absconc_window(dx,recog_tokens,all_tokens,incl_psg=incl_psg,psg_as_markdown=psg_as_markdown,count_keys=count_keys,meta_keys=meta_keys,keyrename=keyrename,num_eg=num_eg,**attrs)
=======
                cdx=_count_absconc_window(dx,recog_tokens,all_tokens,incl_psg=incl_psg,psg_as_markdown=psg_as_markdown,count_keys=count_keys,meta_keys=meta_keys,keyrename=keyrename,vocab_len=vocab_len)
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41
                if cdx:
                    cdx['slice']=len(ld)+1
                    cdx['tok_i']=i+1
                    ld.append(cdx)
                all_tokens,recog_tokens=[],[]
        # keep the last which is shorter?
        if keep_last:
            if all_tokens and recog_tokens:
<<<<<<< HEAD
                cdx=_count_absconc_window(dx,recog_tokens,all_tokens,incl_psg=incl_psg,psg_as_markdown=psg_as_markdown,count_keys=count_keys,meta_keys=meta_keys,keyrename=keyrename,num_eg=num_eg,**attrs)
=======
                cdx=_count_absconc_window(dx,recog_tokens,all_tokens,incl_psg=incl_psg,psg_as_markdown=psg_as_markdown,count_keys=count_keys,meta_keys=meta_keys,keyrename=keyrename,vocab_len=vocab_len)
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41
                if cdx:
                    cdx['slice']=len(ld)+1
                    cdx['tok_i']=i+2
                    ld.append(cdx)
        
    return ld



<<<<<<< HEAD
def count_absconc_corpus(
        cname,
        num_proc=1,
        save=True,
        ofn=None,
        incl_psg=False,
        incl_eg=False,
        eg_keys=['abs','conc','neither'],
        vnum='v9-zcut05',
        **attrs):
=======
def count_absconc_corpus(cname,num_proc=1,save=True,ofn=None,eg_keys=['abs','conc','neither'],sample_n=10,incl_psg=False):
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41

    
    # prepare
    import lltk
    C=lltk.load(cname)
    paths_txt = [t.path_txt.replace('.gz','') for t in C.texts() if os.path.exists(t.path_txt.replace('.gz',''))]
    path2id = dict((t.path_txt,t.id) for t in C.texts())
<<<<<<< HEAD
    # print(paths_txt)
    # execute
    data = pmap_iter(
        count_absconc_path,
        paths_txt,
        kwargs={'incl_psg':incl_psg, 'incl_eg':incl_eg, **attrs},
=======
    # execute
    data = pmap_iter(
        count_absconc_path if not incl_psg else count_absconc_path_psg,
        paths_txt,
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41
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

<<<<<<< HEAD
    if not ofn:
        ofn=f'{COUNT_DIR}/data.absconc.{cname}{".psgs." if incl_psg else "."}{vnum}.jsonl'
    # header = header_absconc if not incl_psg else [h for h in header_absconc if not h in eg_keys]+['passage']
    print('>> writing to:',ofn)
    # print(header_absconc)
    writegen_jsonl(ofn,_gen)
    # save as feather
    #if not incl_psg:
    # pd.read_csv(ofn).reset_index().to_feather(os.path.splitext(ofn)[0]+'.ft.gz')
=======
    if not ofn: ofn=f'{COUNT_DIR}/data.absconc.{cname}{".psgs." if incl_psg else "."}v7.csv.gz'
    header = header_absconc if not incl_psg else [h for h in header_absconc if not h in eg_keys]+['passage']
    print('>> writing to:',ofn)
    writegen(ofn,_gen,header=header)
    # save as feather
    if not incl_psg: pd.read_csv(ofn).reset_index().to_feather(os.path.splitext(ofn)[0]+'.ft')
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41
    
    
    # df=pd.DataFrame(newld)
    # # save?
    # if save:
    #     df.to_csv(ofn,index=False)
    # # return
    # return df
    
<<<<<<< HEAD
=======
PSG_SOURCES={'Median'}
PSG_PERIODS={'median'}
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41

def count_absconc_psg(txt,incl_psg=True,sources=PSG_SOURCES,periods=PSG_PERIODS,**attrs):
    df=pd.DataFrame(count_absconc(txt,incl_psg=incl_psg,sources=sources,periods=periods,window_len=100,**attrs))
    df['abs-conc']=df['num_abs']-df['num_conc']
    return df.sort_values('abs-conc')
<<<<<<< HEAD
    
def count_absconc_json(fnfn,periods=PERIODS_FOR_COUNTING,sources=SOURCES_FOR_COUNTING,modernize=MODERNIZE_SPELLING):
    if not os.path.exists(fnfn): return []
    import ujson
    try:
        with open(fnfn) as f: freqd=ujson.load(f)
    except Exception:
        return []
    if not freqd: return []
    freqdl=Counter()

    if modernize:
        spellingd=get_spelling_modernizer()

    for k,v in freqd.items():
        if not k or not k[0].isalpha(): continue
        kl=k.lower()
        if modernize: kl=spellingd.get(kl,kl)
        freqdl[kl]+=v
    nw=sum(freqdl.values())

    old=[]
    for dx in get_norms_for_counting(sources=sources,periods=periods):
        odx={}
        odx['num_abs'] = sum(freqdl[w] for w in dx['neg'])
        odx['num_conc'] = sum(freqdl[w] for w in dx['pos'])
        odx['num_neither'] = sum(freqdl[w] for w in dx['neither'])
        odx['num_total']=odx['num_abs']+odx['num_conc']+odx['num_neither']
        odx['num_types'] = len([w for w in dx['neg']|dx['pos']|dx['neither'] if w in freqdl and freqdl[w]])
        odx['num_words'] = nw
        odx['source']=dx['source']
        odx['period']=dx['period']
        old.append(odx)
    return old

######
# If all we've got are 
=======
    
>>>>>>> ffd935b4afb7acfab2b22d8a7be044d524564e41
