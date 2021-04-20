from abslithist import *

SOURCES = ['PAV-Conc','MRC-Conc','MT-Conc','PAV-Imag','MRC-Imag','LSN-Imag','LSN-Perc','LSN-Sens','Median']
BAD_SOURCES = {}#'LSN-Perc','LSN-Sens'}



### Funcs
# split semantic axis into high and low fields

# # Save zdata
def add_series_to_norms(series,source,norms,series_std={},**attrs):
    seriesz=zfy(series)
    done=set()
    #for v,z,w in zip(series,seriesz,series.index):
    for w,z in zip(seriesz.index,seriesz):
        if type(w)!=str or not w or not w[0].isalpha(): continue
        wdx={
            'word':w,
            'score':series[w],
            #'std':series_std.get(w),
            'z':z,
            'source':source,
            **attrs
        }
        norms.append(wdx)




def gen_norms_paivio(norms,prefix='PAV'):
    pav_path_pdf=os.path.join(SOURCE_DIR,'Paivio1968.pdf')
    pav_path_csv=os.path.join(SOURCE_DIR,'Paivio1968.csv')
    # generate table from PDF if CSV does not exist
    if not os.path.exists(pav_path_csv):
        from tabula import read_pdf
        dfs_paivio=read_pdf(pav_path_pdf,pages='10-25')
        df_paivio=pd.concat(dfs_paivio)
        pav_header=['Noun','IMAG_M','x','IMAG_SD','CONC_M','y','CONC_SD','MEANP_M','z','MEANP_SD','F']
        df_paivio.columns=pav_header+['a','b','c','d','e','f','g','h','i']
        df_paivio['word']=df_paivio.Noun.str.lower()
        df_paivio[['word','IMAG_M','IMAG_SD','CONC_M','CONC_SD','MEANP_M','MEANP_SD','F']].iloc[2:].to_csv(pav_path_csv,index=False)
    # load csv
    df_paivio=pd.read_csv(pav_path_csv).set_index('word')
    # add to norms
    add_series_to_norms(series=df_paivio.CONC_M,source='Abs-Conc.PAV-Conc',norms=norms,series_std=df_paivio.CONC_SD)
    add_series_to_norms(series=df_paivio.IMAG_M,source='Abs-Conc.PAV-Imag',norms=norms,series_std=df_paivio.IMAG_SD)
    # add to fields

def gen_norms_mrc(norms,prefix='MRC'):
    # urls and paths
    mrc_url_zip='https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/1054/1054.zip?sequence=3&isAllowed=y'
    mrc_path_zip=os.path.join(SOURCE_DIR,'mrc.zip')
    mrc_path_dic=os.path.join(SOURCE_DIR,'mrc2.dct')
    # Download and unpack if necessary
    if not os.path.exists(mrc_path_dic):
        # download zip
        download_tqdm(mrc_url_zip, mrc_path_zip)
        # custom unzip: remove internal directories to unzip directly to txt folder
        from zipfile import ZipFile   
        
        with ZipFile(mrc_path_zip) as zip_file:
            namelist=zip_file.namelist()
            # Loop over each file
            for member in tqdm(iterable=namelist, total=len(namelist)):
                # copy file (taken from zipfile's extract)
                source = zip_file.open(member)
                filename = os.path.basename(member)
                if not filename: continue
                if not os.path.splitext(filename)[1]: continue
                if filename=='mrc2.dct':
                    # write!
                    with open(mrc_path_dic, "wb") as target:
                        with source, target:
                            shutil.copyfileobj(source, target)
                            # print('>> saved:',mrc_path_dic)
            # remove zip
            os.remove(mrc_path_zip)
    
    # Fields we want
    mrc_parser={}
    mrc_parser['CONC']=29,31
    mrc_parser['IMAG']=32,34
    mrc_parser['AOA']=41,43
    mrc_parser['BROWN_FREQ']=22,25
    mrc_parser['FAM']=26,28
    mrc_parser['MEANC']=35,37
    mrc_parser['MEANP']=38,40
    
    # Parse file
    mrc_ld=[]
    with open(mrc_path_dic) as f:
        for i,ln in enumerate(f):#,desc='Parsing MRC...')):
            # print(i,ln)
            # str meta
            dx={}
            dx['ALPHSYL']=ln[46]
            dx['IRREG']=ln[50]
            dx['POS']=ln[44]
            if not dx['POS'] in {'N','J','V','A'}: continue
            # get word
            last=ln.strip().split()[-1]
            word=last.split('|')[0].strip().lower()
            if dx['IRREG'] in {'N'}: word=word[1:]
            dx['word']=word
            # get vals
            for field_name,(start,stop) in mrc_parser.items():
                field_val = ln[start-1:stop]
                field_int = int(field_val)
                if field_int < 100 or field_int>700: field_int=np.nan
                dx[field_name]=field_int
            mrc_ld.append(dx)
    mrc_df=pd.DataFrame(mrc_ld).groupby(['word']).median().reset_index().set_index('word')
    
    # add series's
    add_series_to_norms(series=mrc_df['CONC'], source='Abs-Conc.MRC-Conc', norms=norms)
    add_series_to_norms(series=mrc_df['IMAG'], source='Abs-Conc.MRC-Imag', norms=norms)


## Brysbaert et al
def gen_norms_brys(norms,prefix='MT'):
    # paths
    mturk_url='http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt'
    mturk_path=os.path.join(SOURCE_DIR,'Concreteness_ratings_Brysbaert_et_al_BRM.txt')
    # download if necessary
    if not os.path.exists(mturk_path):
        download_tqdm(mturk_url,mturk_path)
    # load
    df_brys = pd.read_csv(mturk_path,sep='\t').set_index('Word')
    # filter for 1grams
    df_brys['word']=df_brys.index
    df_brys=df_brys[df_brys.word.apply(lambda x: type(x)==str and x and ' ' not in x)]
    add_series_to_norms(series=df_brys['Conc.M'], source='Abs-Conc.MT-Conc', norms=norms, series_std=df_brys['Conc.SD'])

def gen_norms_lsn(norms):
    # paths
    url_lsn_csv='https://osf.io/48wsc/download'
    path_lsn_csv=os.path.join(SOURCE_DIR,'Lancaster_sensorimotor_norms_for_39707_words.csv')
    if not os.path.exists(path_lsn_csv):
        download_tqdm(url_lsn_csv,path_lsn_csv)
    # load from csv
    df_lsn=pd.read_csv(path_lsn_csv)
    df_lsn['Word']=df_lsn.Word.str.lower()
    df_lsn=df_lsn[~df_lsn.Word.str.contains(' ')]
    df_lsn=df_lsn.set_index('Word')
    # add norms
    # add_series_to_norms(series=df_lsn['Minkowski3.perceptual'], source='Abs-Conc.LSN-Perc', norms=norms)
    # add_series_to_norms(series=df_lsn['Minkowski3.sensorimotor'], source='Abs-Conc.LSN-Sens', norms=norms)
    add_series_to_norms(series=df_lsn['Visual.mean'], source='Abs-Conc.LSN-Imag', norms=norms)
    add_series_to_norms(series=df_lsn['Haptic.mean'], source='Abs-Conc.LSN-Hapt', norms=norms)
    # add_series_to_norms(series=df_lsn['Auditory.mean'], source='Abs-Conc.LSN-Aud', norms=norms)


def gen_orignorms():
    # init
    fields=defaultdict(set)
    norms=[]

    # add fields (so far, only quant/scale-based ones)
    funcs=[
        gen_norms_paivio, # Paivo et al
        gen_norms_mrc,    # MRC
        gen_norms_brys,   # Brysbaert et al
        gen_norms_lsn     # LSN
    ]

    # run through functions
    for func in tqdm(funcs,desc='Building fields and norms from sources'):
        func(norms)

    # save norms
    qdf=pd.DataFrame(norms)
    qdf=qdf.drop_duplicates(['word','source'],keep='first')
    qdf=qdf.pivot('word','source','z')   # new!
    qdf.to_csv(PATH_NORMS)#,index=False)
    
def filter_norms(df,remove_stopwords=REMOVE_STOPWORDS_IN_WORDNORMS):
    return df if not remove_stopwords else df.loc[[w for w in df.index if w not in get_stopwords()]]

DATA_ORIGNORMS=None
def get_orignorms(remove_stopwords=REMOVE_STOPWORDS_IN_WORDNORMS):
    global DATA_ORIGNORMS
    if DATA_ORIGNORMS is None:
        # print('loading orig norms')
        df=pd.read_csv(PATH_NORMS).set_index('word')
        df=filter_norms(df,remove_stopwords=remove_stopwords)
        # add median?
        df['Abs-Conc.Median'] = df.median(axis=1)
        DATA_ORIGNORMS=df
    return DATA_ORIGNORMS



#################
# Norms -> Fields
##


def get_fields_from_norms(dfnorms,zcut=ZCUT,neither='Neither',reverse=False,remove_stopwords=True):
    fields=dict()

    if remove_stopwords:
        from abslithist.words import get_stopwords
        dfnorms=dfnorms.loc[set(dfnorms.index)-get_stopwords()]
    
    contrasts = get_contrasts(dfnorms,zcut=zcut,reverse=reverse)
    for cdx in contrasts:
        neg,pos=cdx['contrast'].split('-')
        period='.'+cdx['period'] if cdx['period'] else ''
        fields[f"{cdx['contrast']}.{cdx['source']}.{neg}{period}"]=cdx['neg']
        fields[f"{cdx['contrast']}.{cdx['source']}.{pos}{period}"]=cdx['pos']
        fields[f"{cdx['contrast']}.{cdx['source']}.{neither}{period}"]=cdx['neither']

    return fields

def get_origfields():
    dfnorms=get_orignorms()
    return get_fields_from_norms(dfnorms)

def get_vecfields():
    dfnorms=get_vecnorms()
    return get_fields_from_norms(dfnorms)

def get_allfields():
    return get_fields_from_norms(get_allnorms())

def get_fields(): return get_allfields()

def get_absconc():
    fields = get_allfields()
    return {
        'abs':fields.get(DEFAULT_ABS_FIELD),
        'conc':fields.get(DEFAULT_CONC_FIELD),
        'neither':fields.get(DEFAULT_NEITHER_FIELD),
    }




def sample(l,n=10):
    wordstr=', '.join([str(w) for w in (l if len(l)<n else random.sample(l,n))])
    return f'{wordstr} ... ({len(l)})'

def show_fields(fields):
    ld = [
        {
            'field':field,
            'num':len(fields[field]),
            'words':sample(fields[field])
        } for field in fields
    ]
    pd.options.display.max_colwidth=100
    return pd.DataFrame(ld).set_index('field').sort_values('num',ascending=False)



def show_contrasts(contrasts,sample_n=5,colwidth=50,numrows=100):
    pd.options.display.max_colwidth=colwidth
    pd.options.display.max_rows=numrows
    for dx in contrasts:
        for k in ['pos','neg','neither']:
            dx[k]=sample(dx[k],n=sample_n)
    df=pd.DataFrame(contrasts).sort_values(['contrast','source','period'])
    return df.set_index(['contrast','source','period']).fillna('')
    

def get_contrasts(dfnorms,neither='Neither',reverse=False,zcut=ZCUT):
    ld=[]
    for col in dfnorms.columns:
        colparts=col.split('.')
        contrast=colparts[0]
        source=colparts[1]
        period=colparts[2] if len(colparts)>2 else 'orig'
        neg,pos=contrast.split('-')
        
        series=dfnorms[col]
        # print(series.loc['virtue'])
        poswords=set(series[series>=zcut if not reverse else series<=(zcut*-1)].index)
        negwords=set(series[series<=(zcut*-1) if not reverse else series>=zcut].index)
        neitherwords = set(series.dropna().index) - poswords - negwords


        dx={
            'contrast':contrast,
            'source':source,
            'period':period,
            'neg':negwords, #if not show_sample else sample(fields[negkey]),
            'pos':poswords, #if not show_sample else sample(fields[poskey]),
            'neither':neitherwords,# if not show_sample else sample(fields[neitherkey]),
        }
        ld.append(dx)
    
    return ld

def get_misccontrasts():
    ld=[]
    ld+=[{
        'contrast':'Woman-Man',
        'source':'SingleWords',
        'period':'na',
        'neg':{'man'},
        'pos':{'woman'},
        'neither':{}
    }]
    ld+=[{
        'contrast':'Woman-Man',
        'source':'MultiWords',
        'period':'na',
        'neg':{'man','boy','brother','father','husband','son'},
        'pos':{'woman','girl','sister','mother','wife','daughter'},
        'neither':{}
    }]
    return ld


def get_subcontrasts(remove=REMOVE_STOPWORDS_IN_WORDNORMS, include_misc=True):
    ld=get_origcontrasts()
    if include_misc: ld+=get_misccontrasts()
    return ld


def get_origcontrasts(remove_stopwords=REMOVE_STOPWORDS_IN_WORDNORMS):
    return get_contrasts(get_orignorms(remove_stopwords=remove_stopwords))

def get_veccontrasts(remove_stopwords=REMOVE_STOPWORDS_IN_WORDNORMS):
    return get_contrasts(get_vecnorms(remove_stopwords=remove_stopwords))

def get_allcontrasts(remove_stopwords=REMOVE_STOPWORDS_IN_WORDNORMS,cached=True,cachefn='data/fields/data.cache.allcontrasts.pkl'):
    #if cached and os.path.exists(cachefn):
    #    with open(cachefn,'rb') as f: return pickle.load(f)
    # get otherwise
    ld=get_contrasts(get_allnorms(remove_stopwords=remove_stopwords))
    # save?
    #if cached:
    #    with open(cachefn,'wb') as of: pickle.dump(ld,of)
    return ld

def decideifabs(x,zcut=ZCUT,reverse=False):
    if x>=zcut: return 'Concrete' if not reverse else 'Abstract'
    if x<=(zcut*-1): return 'Abstract' if not reverse else 'Concrete'
    return 'Neither'

def format_norms_as_long(dfnorms,zcut=ZCUT):
    ld=[]
    for col in dfnorms.columns:
        colparts=col.split('.')
        contrast=colparts[0]
        source=colparts[1]
        source_type='Conc' if source.split('-')[-1]=='Conc' else 'Imag'
        period=colparts[2] if len(colparts)>2 else ''
        source=f'{source}.{period}' if period else source
        neg,pos=contrast.split('-')
        
        series=dfnorms[col].dropna()

        for word,z in zip(series.index, series):
            dx={
                'word':word,
                'z':z,
                'source':source.split('.')[0],
                'period':source.split('.')[1],
                # 'source_type':source_type,
                'decision':decideifabs(z),
                'order':SOURCES.index(source) if source in SOURCES else 0
            }
            # if period: dx['period']=period
            ld.append(dx)
    return pd.DataFrame(ld).sort_values('z')
    

ALLNORMS=None
def get_allnorms(remove_stopwords=REMOVE_STOPWORDS_IN_WORDNORMS):
    global ALLNORMS
    if ALLNORMS is None:
        # print('loading allnorms')

        # orig
        dfnorms_orig = get_orignorms(remove_stopwords=remove_stopwords)
        dfnorms_orig.columns = [c+'.orig' for c in dfnorms_orig.columns]
        # vecs
        dfnorms_vec = get_vecnorms(add_median=True,remove_stopwords=remove_stopwords)
        # join
        ALLNORMS=dfnorms_vec.join(dfnorms_orig,how='outer')
    return ALLNORMS

def show_origcontrasts(remove_stopwords=REMOVE_STOPWORDS_IN_WORDNORMS):
    return show_contrasts(get_origcontrasts(remove_stopwords=remove_stopwords))

def show_veccontrasts(remove_stopwords=REMOVE_STOPWORDS_IN_WORDNORMS):
    return show_contrasts(get_veccontrasts(remove_stopwords=remove_stopwords))

def show_allcontrasts(remove_stopwords=REMOVE_STOPWORDS_IN_WORDNORMS):
    return show_contrasts(get_allcontrasts(remove_stopwords=remove_stopwords))



### get scores for set of words
NORMDICTS={}
def get_norm_dict(contrast='Abs-Conc',source='Median',period='median',norms=None,remove_stopwords=True,stopwords=set()):
    global NORMDICTS
    normkey=(contrast,source,period)#,remove_stopwords,stopwords))
    if not normkey in NORMDICTS:
        # print('loading norm dict')
        if norms is None: norms=get_allnorms()
        normsok=norms[f'{contrast}.{source}.{period}'].dropna()
        if not stopwords and remove_stopwords: stopwords=get_stopwords()
        wordsok=set(normsok.index) - stopwords
        w2score=dict((a,b) for a,b in zip(normsok.index, normsok) if a not in stopwords)
        scores=pd.Series(w2score.values())
        w2score_perc=dict((a,percentileofscore(scores,b)) for a,b in w2score.items())
        NORMDICTS[normkey]=(w2score,w2score_perc)
    # else:
        # print('found cached norm dict')
    return NORMDICTS.get(normkey,({},{}))







###############
# VECTOR FIELDS
#

def dist2norms(df):
    norms=[]
    for col in df.columns:
        # add to norms
        add_series_to_norms(
            df[col],
            source=col,
            norms=norms,
        )
    return norms

def gen_vecnorms_for_model(pathd):
    # load model
    import gensim
    from abslithist.models.embeddings import load_model, get_fieldvecs_in_model, compute_vec2vec_dists
    model = load_model(pathd.get('path'),pathd.get('path_vocab'),min_count=MIN_COUNT_MODEL)
    word2vec = dict((w,model[w]) for w in model.wv.vocab)# if model.vocab[w].count>=MIN_COUNT_MODEL)

    # field vectors
    field2vec = get_fieldvecs_in_model(
        model,
        # fields = get_origfields(),   # not getting vectors for non-contrasts for now
        contrasts=get_subcontrasts()
    )

    # dist table
    dfdist = compute_vec2vec_dists(word2vec,field2vec,xname='word',yname='field')

    # normalize
    return dist2norms(dfdist)


def gen_vecnorms_for_paths(paths,desc=None,num_proc=1):
    norms = [
        normd
        for normld in pmap(
            gen_vecnorms_for_model,
            paths,
            num_proc=num_proc,
            desc=desc,
            use_threads=False
        ) for normd in normld
    ]
    dfnorms=pd.DataFrame(norms).groupby(['word','source']).median().reset_index()
    return dfnorms




def gen_vecnorms(bin_year_by=MODEL_PERIOD_LEN,num_runs=None,num_proc=1):
    """
    Aggregate model-periods' vecnorms by century/yearbin
    """
    from abslithist.models.embeddings import get_model_paths

    def periodize(y):
        y=int(y)
        if bin_year_by==100:
            return f'C{(y//100) + 1}'
        elif bin_year_by==50:
            return f'C{(y//100) + 1}{"e" if int(str(y)[2])<5 else "l"}'
        else:
            return y//bin_year_by * bin_year_by

    # group by paths    
    paths_ld = get_model_paths()
    paths_df = pd.DataFrame(paths_ld)
    paths_df['period'] = paths_df['period_start'].apply(periodize)
    # print(len(paths_df))

    ## agg by period
    # alldf=pd.DataFrame()
    word2field2z=defaultdict(dict)
    for period,perioddf in sorted(paths_df.groupby('period')):
        ofn_period=os.path.splitext(PATH_VECNORMS)[0]+'.'+period+'.csv.gz'
        if not os.path.exists(ofn_period):
            newdf=pd.DataFrame()
            for (corpus,period_start,period_end),cppdf in sorted(perioddf.groupby(['corpus','period_start','period_end'])):
                # print(period,corpus,period_start,period_end)
                # newdf=newdf.append(
                corpusdf=gen_vecnorms_for_paths(
                        cppdf.sort_values('path').to_dict('records')[:num_runs],
                        desc=f'Computing norms for {period} ({corpus}, {period_start}-{period_end})',
                        num_proc=num_proc
                )
                corpusdf['corpus']=corpus
                # break
                newdf=newdf.append(corpusdf)
            newdf=newdf.groupby(['word','source','corpus']).median().reset_index()
            newdf=newdf.groupby(['word','source']).median().reset_index()
            for word,field,z in zip(newdf.word, newdf.source, newdf.z):
                if not word or not word[0].isalpha(): continue
                newfield=field+'.'+period
                word2field2z[word][newfield]=z
        # break

    ld = [
        {
            **{'word':word},
            **word2field2z[word]
        } for word in word2field2z
    ]
    df=pd.DataFrame(ld).set_index('word')
    df.to_csv(PATH_VECNORMS)


def get_vecnorms_fns(periods=None):
    l=[]
    for fn in sorted(os.listdir(FIELD_DIR)):
        if not fn.startswith(VECNORMS_FN_PRE): continue
        period=fn.replace('.gz','').replace('.csv','').split('.')[-1]
        if periods and period not in periods: continue
        l.append((period,os.path.join(FIELD_DIR,fn)))
    return l

DATA_VECNORMS=None
def get_vecnorms(periods=None,add_median=True,remove_stopwords=REMOVE_STOPWORDS_IN_WORDNORMS):
    global DATA_VECNORMS
    if DATA_VECNORMS is None:
        # print('loading vec norms')
        df=pd.read_csv(PATH_VECNORMS).set_index('word')
        df=filter_norms(df,remove_stopwords=remove_stopwords)
        
        colgroups=defaultdict(set)
        for col in df.columns:
            if col.count('.')!=2: continue
            contrast,source,period=col.split('.')#[-1]
            colgroups[contrast+'.'+source]|={col}
        
        for contrastsource,pcols in colgroups.items():
            dfp = df[pcols].median(axis=1)
            newcolname=contrastsource+'.median'
            df[newcolname]=dfp
        DATA_VECNORMS=df
    return DATA_VECNORMS


    # periods = set(col.split('.')[-1] for col in df.columns if col.count('.')>1)
    # for period in periods:
    #     pcols=[col for col ]

    
    # for period,fnfn in get_vecnorms_fns(periods=periods):
    #     df=pd.read_csv(fnfn)

    #     # HACK!! THIS WAS OWING TO CHANGE IN norms->fields function in between generating data. fix!
    #     if period.startswith('C16'):
    #         df['score']*=-1
    #         df['z']*=-1
        
    #     yield (period,df)

# def gen_vecfields(periods=None):
#     fields={}
#     total=len(get_vecnorms_fns(periods=periods))
#     for period,perioddf in tqdm(get_vecnorms(),total=total,desc='Splitting into fields'):
#         period_fields=get_fields_from_norms(perioddf)
#         for field,words in sorted(period_fields.items()):
#             newfield=f'{field}.{period}'
#             print(newfield,len(words),random.sample(words,5))
#             fields[newfield]=words
    
#     # save
#     with open(PATH_VECFIELDS_PKL,'wb') as of:
#         pickle.dump(fields,of)


def gen_vecfields(periods=None):
    word2field=defaultdict(dict)
    total=len(get_vecnorms_fns(periods=periods))
    for period,perioddf in tqdm(get_vecnorms(),total=total,desc='Splitting into fields'):
        period_fields=get_fields_from_norms(perioddf,remove_stopwords=False)
        for field,words in sorted(period_fields.items()):
            newfield=f'{field}.{period}'
            print(newfield,len(words),random.sample(words,5))
            #fields[newfield]=words
            for word in words:
                word2field[word][newfield]='y'
        # break
    
    ld = [
        {
            **{'word':word},
            **word2field[word]
        } for word in word2field
    ]
    df=pd.DataFrame(ld).set_index('word')
    df.to_csv(PATH_VECFIELDS)




#### Stats

def corr_norms(dfnorms):
    from scipy.stats import pearsonr
    def pearsonr_pval(x,y): return pearsonr(x,y)[1]
    pd.options.display.max_rows=10
    cordf_r=dfnorms.corr()
    cordf_p=dfnorms.corr(method=pearsonr_pval)
    cordf_r_melt=cordf_r.reset_index().melt('index').set_index(['index','variable'])
    cordf_p_melt=cordf_p.reset_index().melt('index').set_index(['index','variable'])
    cordf=cordf_r_melt.join(cordf_p_melt,rsuffix='_p')
    cordf=cordf.reset_index()
    cordf=cordf[cordf['index']<cordf['variable']]#.set_index(['inde'])
    cordf=cordf.sort_values('value')
    print('Minimum:',cordf['value'].min())
    print('Median:',cordf['value'].median())
    print('Maximum:',cordf['value'].max())
    return cordf

# def corr_norms2(dfnorms):
#     from scipy.stats import pearsonr
#     pd.options.display.max_rows=10

#     ld=[]
#     return dfnorms
#     for col1 in dfnorms.columns:
#         for col2 in dfnorms.columns:
#             if col1>=col2: continue
#             s1=dfnorms[col1].replace([np.inf, -np.inf], np.nan).dropna()
#             s2=dfnorms[col2].replace([np.inf, -np.inf], np.nan).dropna()
#             shared=set(s1.index)&set(s2.index)
#             # print(s1.loc[shared],s2.loc[shared])
#             s1x=s1.loc[shared]
#             s2x=s2.loc[shared]
#             #print(set(s2x.index)-set(s1x.index))
#             #print(col1,col2,len(s1),len(s2),len(shared),len(s1x),len(s2x))#,shared)

#             try:
#                 r,p=pearsonr(s1x.values,s2x.values)
#             except ValueError:
#                 continue
#             dx={'col1':col1,'col2':col2,'r':r,'p':p,'n':len(shared)}
#             ld.append(dx)
#     cordf=pd.DataFrame(ld)
#     print('Minimum:',cordf['r'].min())
#     print('Median:',cordf['r'].median())
#     print('Maximum:',cordf['r'].max())
#     return cordf.sort_values('r')


def corr_orignorms(): return corr_norms(get_orignorms())
def corr_vecnorms(): return corr_norms(get_vecnorms())
def corr_allnorms(): return corr_norms(get_allnorms())