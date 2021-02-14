import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..'))
from abslithist import *
SOURCE_DIR=os.path.join(PATH_DATA,'fields','sources')
FIELD_DIR=os.path.join(PATH_DATA,'fields')
MODEL_DIR=os.path.join(PATH_DATA,'models')
PATH_FIELDS_JSON = os.path.join(FIELD_DIR,'data.fields_orig.json')
PATH_VECFIELDS_JSON = os.path.join(FIELD_DIR,'data.fields_vec.json')
PATH_FIELD2VEC_PKL = os.path.join(FIELD_DIR,'data.models.word_and_field_vecs.pkl')
PATH_NORMS = os.path.join(FIELD_DIR,'data.wordnorms_orig.csv')
PATH_VECNORMS = os.path.join(FIELD_DIR,'data.wordnorms_vec.csv')
if not os.path.exists(SOURCE_DIR): os.makedirs(SOURCE_DIR)
ZCUT = ZCUT_NORMS_ORIG

### Funcs
# split semantic axis into high and low fields
def split_spectrum_into_fields(series):
    series_z=zfy(series)
    top = set(series_z[series_z>=ZCUT].index)
    bottom = set(series_z[series_z<=-ZCUT].index)
    middle = set(series_z.index) - top - bottom
    return (bottom,middle,top)

def add_series_to_fields(series,fields,method,contrast='Abs-Conc',contrast_vals=['Abs','Neither','Conc'],suffix=''):
    (
        fields[f'{contrast}.{method}.{contrast_vals[0]}{suffix}'],
        fields[f'{contrast}.{method}.{contrast_vals[1]}{suffix}'],
        fields[f'{contrast}.{method}.{contrast_vals[2]}{suffix}']
    ) = split_spectrum_into_fields(series)

# # Save zdata
def add_series_to_norms(series,source,norms,series_std={},**attrs):
    seriesz=zfy(series)
    done=set()
    for v,z,w in zip(series,seriesz,series.index):
        wdx={
            'word':w,
            'score':v,
            #'std':series_std.get(w),
            'z':z,
            'source':source,
            **attrs
        }
        norms.append(wdx)




def add_fields_paivio(fields,norms,prefix='PAV'):
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
    add_series_to_norms(series=df_paivio.CONC_M,source='PAV-Conc',norms=norms,series_std=df_paivio.CONC_SD)
    add_series_to_norms(series=df_paivio.IMAG_M,source='PAV-Imag',norms=norms,series_std=df_paivio.IMAG_SD)
    # add to fields
    add_series_to_fields(series=df_paivio.CONC_M,fields=fields,method='PAV-Conc')
    add_series_to_fields(series=df_paivio.IMAG_M,fields=fields,method='PAV-Imag')

def add_fields_mrc(fields,norms,prefix='MRC'):
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
    add_series_to_norms(series=mrc_df['CONC'], source='MRC-Conc', norms=norms)
    add_series_to_norms(series=mrc_df['IMAG'], source='MRC-Imag', norms=norms)

    # add fields
    add_series_to_fields(series=mrc_df.CONC,fields=fields,method='MRC-Conc')
    add_series_to_fields(series=mrc_df.IMAG,fields=fields,method='MRC-Imag')
    add_series_to_fields(series=mrc_df.AOA,fields=fields,method='MRC-AOA',contrast='Early-Late',contrast_vals=['Early','Neither','Late'])


## Brysbaert et al
def add_fields_brys(fields,norms,prefix='MT'):
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
    df_brys=df_brys[df_brys.word.apply(lambda x: type(x)==str and ' ' not in x)]
    # add series
    add_series_to_norms(series=df_brys['Conc.M'], source='MT-Conc', norms=norms, series_std=df_brys['Conc.SD'])
    # add fields
    add_series_to_fields(series=df_brys['Conc.M'],fields=fields,method='MT-Conc')

def add_fields_lsn(fields,norms):
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
    add_series_to_norms(series=df_lsn['Minkowski3.perceptual'], source='LSN-Perc', norms=norms)
    add_series_to_norms(series=df_lsn['Minkowski3.sensorimotor'], source='LSN-Sens', norms=norms)
    add_series_to_norms(series=df_lsn['Minkowski3.action'], source='LSN-Act', norms=norms)
    add_series_to_norms(series=df_lsn['Visual.mean'], source='LSN-Imag', norms=norms)
    # add to fields
    add_series_to_fields(df_lsn['Visual.mean'],fields=fields,method='LSN-Imag')
    add_series_to_fields(df_lsn['Minkowski3.perceptual'],fields=fields,method='LSN-Perc')
    add_series_to_fields(df_lsn['Minkowski3.action'],fields=fields,method='LSN-Act')
    add_series_to_fields(df_lsn['Minkowski3.sensorimotor'],fields=fields,method='LSN-Sens')



def gen_fields_and_norms():
    # init
    fields=defaultdict(set)
    norms=[]

    # add fields (so far, only quant/scale-based ones)
    funcs=[
        add_fields_paivio, # Paivo et al
        add_fields_mrc,    # MRC
        add_fields_brys,   # Brysbaert et al
        add_fields_lsn     # LSN
    ]

    # run through functions
    for func in tqdm(funcs,desc='Building fields and norms from sources'):
        func(fields,norms)

    # save fields
    fields_save=dict((k,list(v)) for k,v in fields.items())
    with open(PATH_FIELDS_JSON,'w') as of: json.dump(fields_save, of)
    
    # save norms
    qdf=pd.DataFrame(norms)
    qdf=qdf.drop_duplicates(['word','source'],keep='first')
    qdf.to_csv(PATH_NORMS,index=False)
    

def get_origfields():
    with open(PATH_FIELDS_JSON) as f: d=json.load(f)
    for k,v in d.items(): d[k]=set(v)
    return d


### VEC FIELDS
def _calc_norms_dist_group(obj):
    path_ld_group=obj
    fields=get_origfields()

    # get fn
    pathd=path_ld_group[0]
    ofn_norms=os.path.join(PATH_MODELS,pathd['corpus'],pathd['period_start']+'-'+pathd['period_end'],'data.wordnorms_vec.csv')
    if os.path.exists(ofn_norms): return

    # loop
    norms=[]
    for pathd in path_ld_group:
        path=pathd['path']
        df=pd.read_csv(path).set_index('word')
        for col in df.columns:
            # add to norms
            add_series_to_norms(
                df[col],
                source=col,
                norms=norms,
            )
            # add to fields
    dfnorms=pd.DataFrame(norms).groupby(['word','source']).median().reset_index()
    dfnorms.to_csv(ofn_norms,index=False)       
    

def gen_vecnorms():
    # paths
    from abslithist.models.embeddings import get_model_paths
    paths_ld = get_model_paths(model_fn='word2field_dists.csv')
    paths_df = pd.DataFrame(paths_ld)

    # get fields
    # fields = get_origfields()

    # group runs
    groups=[
        group_df.to_dict('records')
        for gi,group_df in paths_df.groupby(['corpus','period_start','period_end'])
    ]

    pmap(_calc_norms_dist_group,groups,desc='Generating norms across model-periods',num_proc=1)
        

         






if __name__=='__main__':
    #gen_fields_and_norms()
    gen_vecfields(compute=False)