import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..'))
from abslithist import *
from abslithist.words import *

FICTION_CORPUS_NAME='CanonFiction'

def binz(z,zcut=1):
    if z>=zcut: return 'Abs'
    if z<=(-1 * zcut): return 'Conc'
    return 'Neither'
def biny(y,by=100,miny=1500,offy=1400):
    return y//by*by if y>=miny else offy

def binz2(row,zcut=1,zcut_both=0):
    zbin=binz(row['abs-conc_z'],zcut=zcut)
    if zbin == 'Neither':
        if row['num_abs_z']>=zcut_both and row['num_conc_z']>=zcut_both:
            zbin='Both'
    return zbin

def get_all_passages(cname=FICTION_CORPUS_NAME):
    df=pd.read_csv(os.path.join(COUNT_DIR,f'data.absconc.{cname}.psgs.v5.csv.gz'))
    df['abs-conc']=df['num_abs']-df['num_conc']
    # df['abs+conc']=df['num_abs']+df['num_conc']
    # df['abs/conc']=df['num_abs']/df['num_conc']
    
    for k in ['abs-conc','num_abs','num_conc','num_neither']:
        df[f'{k}_z']=zscore(df[k])

    df['zbin']=df.apply(binz2,1)
    # get year
    C=lltk.load(cname)
    meta=C.metadata
    id2year=dict(zip(meta.id,meta.year))
    df['year']=[id2year.get(idx) for idx in df.id]
    df['ybin']=df['year'].apply(biny)
    return df

    #C=lltk.load(cname)
    #meta=C.metadata
    #df=meta[['id','author','title','year','major_genre','canon_genre']].merge(df,on='id').sort_values('abs-conc')

def sample_passages(df,sample_by=['ybin','zbin'],n=100):
    df_sample=df[df.zbin!=None].groupby(sample_by).sample(n=n,replace=True).drop_duplicates().sample(frac=1)
    return df_sample

# convert to prodigy

def to_prodigy(df_sample,ofn,force=False):
    # do not allow overwrite
    ofn=os.path.join(PSGS_DIR,ofn)
    if not force and os.path.exists(ofn):
        print(f'{ofn} already exists!')
        return
    from bs4 import BeautifulSoup
    with open(ofn,'w') as of:
        for d in tqdm(df_sample.to_dict('records')):
            text=d['passage']
            dom=BeautifulSoup(text,'lxml')
            d2={'text':dom.get_text(), 'html':d['passage']}
            d2['meta']=dict((k,v) for k,v in d.items() if k!='passage')
            d2str=json.dumps(d2)
            of.write(d2str+'\n')


def printpsg(row):
    printm(row.passage)
    try:
        printm(f'-- {row.author}, <i>{row.title}</i> ({row.year}) [abs-conc={row["abs-conc"]}; abs-conc_z={round(row["abs-conc_z"],2)}]')
        printm(f'tags: {", ".join(row.tags)}')
    except AttributeError:
        pass