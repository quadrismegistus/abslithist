from abslithist import *
VERSION='v6'
FN_BIGGERDATA=os.path.join(PATH_SCORES, 'data.scores.biggerdata.v3.pkl')
FN_BIGHIST=os.path.join(PATH_SCORES, 'bytext5', 'BigHist.pkl')
# datadir0=os.path.join(PATH_DATA,'scores','v4')
# datadir=os.path.join(PATH_DATA,'scores','v5')

# def get_fns():
#     fns=[]
#     fns+=[os.path.join(datadir0,fn,'cache') for fn in os.listdir(datadir0)]
#     fns+=[os.path.join(datadir,fn,'cache') for fn in os.listdir(datadir)]
#     return fns
# # func


badcorps={'COCA','FanFic','PMLA'}

def titlegenre(title):
    twords=set(tokenize(title))
    
    keywords=[('Romance', {'romance', 'romances'}),
 ('Adventure', {'adventure', 'adventures'}),
 ('Tale', {'tale', 'tales'}),
 ('Novel', {'novel', 'novella', 'novels'}),
 ('Essay', {'essay', 'essays'}),
 ('History', {'histories', 'history'}),
 ('Sermon', {'sermon', 'sermons'}),
 ('Letters', {'letter', 'letters'}),
 ('Treatise', {'treatise', 'treatises'}),
 ('Discourse', {'discourse', 'discourses'}),
#  ('Fiction', {'rogue'})
]
    for g,gw in keywords:
        if twords&gw:
            return g
    return None

def getgenre(row):
    if row.genre and row.genre not in {'Prose','Essay','Treatise','Print','Unknown',''}:
        return row.genre
    return titlegenre(row.title)

def fix_hathi_id(x):
    return x.split('/',1)[0] + '/' + x.split('/',1)[-1].replace('/','')




# def get_all_text_scores(cols=['id','title','year','genre','genre2','medium','corpus','canon_genre','major_genre','subcorpus'],force=False,corpora=[]):
#     if os.path.exists(FN_BIGGERDATA) and not force: return lltk.read_df(FN_BIGGERDATA)
#     import mpi_slingshot as sl
#     fns=sorted([
#         fn
#         for fn in get_fns()
#         if fn.split('/')[-2] not in badcorps
#         and (not corpora or fn.split('/')[-2] in corpora)
#     ])
#     dffns=pd.DataFrame({'path_freqs':get_fns()})
#     dffns['corpus']=dffns.path_freqs.apply(lambda x: x.split('/')[-2])
#     dffns=dffns.drop_duplicates('corpus',keep='last')
#     dffns=dffns[~dffns.corpus.isin(badcorps)]
#     if corpora: dffns=dffns[dffns.corpus.isin(corpora)]
#     dffns=dffns.sort_values('corpus')
#     dfs=[]
#     iterr=tqdm(list(range(len(dffns))))
#     for i in iterr:
#         row=dffns.iloc[i]
#         cname,fn=row.corpus, row.path_freqs
# #         print(cname)

#         iterr.set_description(cname)
#         try:
#             df=pd.DataFrame(y for x,y in sl.stream_results(fn,progress=False) if type(y)==dict)
#             df=df[[c for c in df.columns if c!='path_freqs']].melt('id')
#             df['contrast'],df['source'],df['period']=zip(*[x.split('.') for x in df.variable])
#             df=df.drop('variable',1)
# #             if cname=='HathiEngLit':
# #                 df['id']=df['id'].apply(fix_hathi_id)
# #             display(df)
#             dfmeta=lltk.load(cname).meta.reset_index()
#         except KeyError as e:
#             print('!',e)
#             continue
# #         print(len(df),len(dfmeta))
#         if not len(df):
#             print('Skipping',fn)
#             continue
#         dfmeta['corpus']=cname
#         for c in cols: 
#             if not c in dfmeta.columns:
#                 dfmeta[c]=''
#         dfmeta['genre2']=dfmeta.apply(getgenre,axis=1)
#         dfmeta['year']=pd.to_numeric(dfmeta['year'],errors='coerce')
        
        
# #         display(dfmeta)
#         cols2=[c for c in cols if c in set(dfmeta.columns)]
# #         print(dfmeta.columns)
# #         print(df.columns)
#         try:
#             dfm1=dfmeta[cols2].merge(df,on='id',how='inner')
# #             display(dfm1)
#             dfm=dfm1.groupby(cols2+['period','source']).mean().reset_index()
#         except Exception as e:
#             print('???',e)
# #             stop
#             continue
# #         display(dfm)
#         dfs.append(dfm)
# #         break
#     DF=pd.concat(dfs)
#     lltk.save_df(DF,FN_BIGGERDATA)
#     print('>> saved:',FN_BIGGERDATA)
#     return DF



# def get_all_text_scores(
#         incl_meta=[
#             'id',
#             'title',
#             'year',
#             'genre',
#             'genre2',
#             'medium',
#             'corpus',
#             'canon_genre',
#             'major_genre',
#             'subcorpus'
#         ],
#         force=False,
#         corpora=[]):
    
#     dfs=[]
#     for fn in os.listdir(PATH_SCOREDATNOW):
#         if fn.endswith('.pkl'):
#             print(fn)
#             try:
#                 dfs+=[pd.read_pickle(os.path.join(PATH_SCOREDATNOW, fn))]
#             except ValueError:
#                 pass
#     df=pd.concat(dfs)
#     if corpora: df=df[df.corpus.isin(set(corpora))]
#     dfm=pd.DataFrame(yield_corpora_meta(set(df.corpus), incl_meta=incl_meta)).reset_index()
#     odf=df.merge(dfm,on=['corpus','id'])
#     return odf

def get_all_text_scores(corpora=corpora_get_all_text_scores,incl_meta=incl_meta_all_text_scores):
    df=pd.concat(score_corpus(c,freqs_only=True) for c in tqdm(corpora))
    dfm=lltk.small_meta(corpora,incl_meta=incl_meta)
    return df.merge(dfm,on=['corpus','id']).query('tok_val!=0')

def to_scores(sentdf_or_txt,norms=None,source='Median',period='median',tokname='tokl_mod',valname='val',nmin=50,stopwords=set(),sep_para='\n\n',w2score=None,w2score_perc=None):
    # parse sents
    sentdf = to_sents(sentdf_or_txt,sep_para=sep_para) if type(sentdf_or_txt)==str else sentdf_or_txt

    # get norms
    if w2score is None or w2score_perc is None:
        w2score,w2score_perc=get_norm_dict(norms=norms,source=source,period=period,stopwords=stopwords,remove_stopwords=True)

    # set value
    sentdf[valname]=sentdf[tokname].apply(lambda x: w2score.get(x))
    sentdf[valname+'_perc']=sentdf[tokname].apply(lambda x: w2score_perc.get(x))
    sentdf['is_recog']=sentdf[valname].apply(lambda x: isnan(x))
    return sentdf#.set_index('i_tok')


def score_freqs(freqd_or_path_freqs,w2score,w2score_perc={}):
    # print('computing score freqs')
    if type(freqd_or_path_freqs)==str: 
        with open(freqd_or_path_freqs) as f: freqd=json.load(f)
    else:
        freqd=freqd_or_path_freqs
    freqd2=Counter()
    for w,c in freqd.items(): freqd2[w.lower().strip()]+=c
    freqd2=dict((w,c) for w,c in freqd2.items() if w in w2score)
    summ=sum(freqd2.values())
    tf = dict((w,c/summ) for w,c in freqd2.items())
    odx={}
    odx['val']=sum((tf.get(w,0) * w2score.get(w,0)) for w in tf)
    odx['val_perc']=sum((tf.get(w,0) * w2score_perc.get(w,0)) for w in tf)
    # print('done')
    return odx

def score_corpus(C,force=False,num_proc=DEFAULT_NUM_PROC,nmin=50,freqs_only=True,genres={},period=None):
    # objs
    C=lltk.load(C) if type(C)==str else C
    ofn=os.path.join(PATH_SCOREDATNOW,C.name+'.pkl')
#     print(ofn, os.path.exists(ofn))
    if not force and os.path.exists(ofn): return read_df(ofn)
    
    objs = [
        (
            dx.get('path_txt'),
            dx.get('path_freqs'),
            os.path.join(PATH_SCORES_BYTEXT,dx.get('corpus'),dx.get('id')),
            to_field_period(dx.get('year')) if not period else periode,
            {'id':dx.get('id'), 'corpus':dx.get('corpus')},
            1,#num_proc,
            nmin,
            freqs_only
        ) for dx in C.meta_iter()
        if 'id' in dx
        and 'corpus' in dx
        and ('path_txt' in dx or 'path_freqs' in dx)
        and dx.get('year')
        and (not genres or dx.get('genre') in genres)
    ]
    print(len(objs))

    # Do all
    res=pd.DataFrame(pmap(
        do_score_text,
        objs,
        num_proc=num_proc,
        desc='Scoring passages',
    ))
    
    # clean
    res.groupby('id').mean()
    
    save_df(res, ofn)
    return res


def do_score_text(inp):
    path_txt,path_freqs,odir,period,ometa,num_proc,nmin,freqs_only=inp
    ofn_scores = os.path.join(odir,'scores.pkl')
    ofn_psgs = os.path.join(odir,'passages.pkl')
    ofn_freqscore = os.path.join(odir,'freqscore.pkl')
    ensure_dir_exists(odir)
    odx={**ometa}
    # load
    w2score,w2score_perc=get_norm_dict(period=period)
    if not freqs_only and os.path.exists(path_txt):
        try:
            # gen scores
            if os.path.exists(ofn_scores):
                # print('loading',ofn_scores)
                scoredf=pd.read_pickle(ofn_scores)
            else:
                with open(path_txt) as f: txt=f.read()
                scoredf = to_scores(txt,sep_para=None,period=period,w2score=w2score,w2score_perc=w2score_perc)
                scoredf.to_pickle(ofn_scores)
            
            # gen psgs
            if os.path.exists(ofn_psgs):
                # print('loading',ofn_psgs)
                psgdf=pd.read_pickle(ofn_psgs)
                # print('done')
            else:
                # print('??',ofn_psgs)
                psgdf=to_passages(scoredf,num_proc=num_proc,progress=num_proc>1)
                psgdf.to_pickle(ofn_psgs)    

            # get stats
            # psgdx=dict(psgdf[psgdf.num_recog>=nmin][[x for x in psgdf.columns if x.startswith('val')]].mean())
            # scoredx=dict(scoredf[[x for x in scoredf.columns if x.startswith('val')]].mean())
            psgdx=dict(psgdf[psgdf.num_recog>=nmin][[x for x in psgdf.columns if not x.startswith('i_')]].mean())
            psgdx['len']=len(psgdf[psgdf.num_recog>=nmin])
            # print(psgdx)
            #scoredx=dict(scoredf[[x for x in scoredf.columns if not x.startswith('i_')]].mean())
            scoredx={}
            # print('first mean')
            scoredx['val']=scoredf['val'].mean()
            # print('second mean')
            scoredx['val_perc']=scoredf['val_perc'].mean()
            scoredx['len_recog']=scoredf.is_recog.sum()
            scoredx['len']=len(scoredf) - scoredf.is_punct.sum()
            # print(scoredx)
            for k,v in scoredx.items(): odx[f'tok_{k}']=v
            for k,v in psgdx.items(): odx[f'psg_{k}']=v
        except KeyError:
            pass
        
    elif os.path.exists(path_freqs):
        try:
            freqdx = score_freqs(path_freqs,w2score,w2score_perc)
            # with open(ofn_freqscore,'wb') as of: pickle.dump(freqdx,of)
            for k,v in freqdx.items(): odx[f'tok_{k}']=v
        except Exception:
            pass

    # print('returning',odx)
    return odx




def get_current_text_scores(corpora=[DEFAULT_CORPUS], incl_meta=['author','title','year','canon_genre']):
    ifns=[os.path.join(PATH_SCORES_BYTEXT,cname+'.pkl') for cname in corpora]
    df=pd.concat(pd.read_pickle(ifn) for ifn in ifns)
    df['val']=df[[x for x in df.columns if x.endswith('_val')]].mean(axis=1)
    df['val_perc']=df[[x for x in df.columns if x.endswith('_val_perc')]].mean(axis=1)
    #df=df.set_index(['corpus','id'])
    
    # join with meta?
    dfmeta=corpora_meta(corpora,incl_meta=incl_meta)
    odf=df.join(dfmeta)
    odf=odf.dropna().sort_values('val')
    return odf


def get_current_text_scores_canon(df0=None,incl_meta=['year','author','canon_genre','major_genre','canon_name','subcorpus'],only_fic=True):
    from .passages import get_current_psg_scores
    df=get_current_psg_scores(incl_meta=incl_meta) if df0 is None else df0
    if only_fic: df=df[~df.major_genre.isin({'Other','Epic','Verse','Drama','Dialogue','History'})]
    df=df.dropna()
    df['num_psg']=1
    df = df.groupby('id').agg(dict(
        (c,np.mean if not c.startswith('num_') else np.sum)
        for c in df.select_dtypes('number').columns
    ))
    df=df[['val']+[c for c in df.columns if c !='val']]
    
    # reattach
    df=df.drop('year',1).join(lltk.load('CanonFiction').meta)

    # filter out americans?
    df=df[~df.subcorpus.str.contains('American')]
    
    return df.sort_values('val')


def get_current_canon_genre_scores(
        psgdf=None,
        metadf=None,
        genres=None,
        periods=None,
    **attrs):
    
    if psgdf is None: psgdf=get_current_psg_scores(**attrs)
    if metadf is None: metadf=lltk.load_corpus('CanonFiction').meta
    meta=metadf
        
    
    # filter genres
    if not genres: genres=[
    'Epic',
    'Novel',
    'Romance',
    'Picaresque'
    ]
    def togenre(x):
        if x=='Novella': return 'Novel'
        if x in {'Novel','Romance','Picaresque','Epic'}: return x
        if x in {'Tale'}: return 'Romance'
        return None
    meta['Genre']=pd.Categorical(meta['major_genre'].apply(togenre),categories=genres)
    meta['Genre2']=pd.Categorical(meta['canon_genre'])

    # filter periods
    if not periods: periods=[
#         'Classical',
#         'Medieval-C16',
        'Cl.',
        'Med.',
        'C17',
        'C18',
        'C19',
        'C20',
        'C21'
    ]
    def toperiod(y):
        if y<400: return 'Cl.'
#         if y<1600: return 'Medieval-C16'
        if y<1600: return 'Med.'
        return to_cent(y)
    meta['Period']=pd.Categorical(meta['year'].apply(toperiod),categories=periods)

    # combine into one col
    meta=meta[~meta.Period.isnull()]
    meta=meta[~meta.Genre.isnull()]
    meta['Period_Genre']=[
        f'{a} {b}'
        for a,b in zip(meta.Period,meta.Genre)
    ]
    
    # final filters for texts
    textdf = meta[~meta.Genre.isnull() & ~meta.Period.isnull()].reset_index()
    textdf=textdf[~textdf.id.str.contains('Odyssey') | textdf.id.str.contains('Chapman')]
    textdf=textdf[textdf.subcorpus!='Early_American_Fiction']
    textdf=textdf[~textdf.id.str.contains('Amadis') | textdf.id.str.contains('Book 1')]
    textdf=textdf.groupby('Period_Genre').filter(lambda g: len(g)>=3)

    # genre/period filter
    pgs=[
    'Cl. Epic',
    'Cl. Novel',
    'Med. Epic',
    'Med. Romance',
    'Med. Picaresque',
    'C17 Romance',
    'C17 Novel',
    'C17 Picaresque',
    'C18 Romance',
    'C18 Novel',
    'C19 Novel',
    'C20 Novel',
    'C21 Novel'
    ]
    textdf['Period_Genre']=pd.Categorical(textdf['Period_Genre'], categories=list(pgs))

    
    # get psgscore too
    psgscoredf = textdf.merge(psgdf,on='id',how='inner',suffixes=['',2])
    
    # text scores
    scoredf=psgdf.groupby('id').mean()
    textscoredf = textdf.merge(scoredf,on='id',how='inner',suffixes=['','2'])

    return (textscoredf,psgscoredf)