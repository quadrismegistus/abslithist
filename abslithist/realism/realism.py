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



def get_current_text_scores():
    return pd.read_csv(PATH_SCORE_CURRENT)

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
    meta=C.metadata.reset_index()
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
    # printm(row.passage)
        
    psg=row.passage.replace('\\\\','\n')
    while '\n\n' in psg: psg=psg.replace('\n\n','\n')
    psg=psg.strip().replace('\n','\n>\t')
    psg=psg.replace("''",'"')
    unit = f"""
> ... {psg} ...
> 
> -- {row.get("author","Unknown Author")}, _{row.get("title","Unknown Title")}_ ({row.get("year","Unknown Year")})
>    - Abstract words ({row['num_abs']})
>        - {row['abs']}
>    - Concrete words ({row['num_conc']})
>        - {row['conc']}
>    - Neither ({row['num_neither']})
>        - {row['neither']}
>    - Abs - Conc = {row['abs-conc']} ({row.get("abs-conc_z","?")}z)
>    - Type / Token = {int(round(row['num_types'] / row['num_tokens'] * 100,0))}
"""
    printm(unit)




def gen_bookpassages(t_or_idx,corpus=None,fname=None,save=False,periods={'median'},sources={'Median'}):
    import lltk
    if type(t_or_idx)==str:
        if not corpus: return
        C=lltk.load(corpus) if type(corpus)==str else corpus
        if not idx in C.textd: return
    else:
        t=t_or_idx

    ld=count_absconc_path(
        t.path_txt,
        sources=sources,
        periods=periods,
        incl_psg=True,
        incl_eg=True,
        num_eg=25,
        psg_as_markdown=True,
        markdown_uses_html=False,
        modernize=True,
        progress=True
    )
    df=pd.DataFrame(ld)
    df['abs-conc']=df['num_abs']-df['num_conc']
    df['id']=t.id
    df=df.merge(t.corpus.metadata,on='id',how='left')
    if save: save_bookpassages(df,fname=fname if fname else t.id)
    return df

def save_bookpassages(df,fname,stacklen=100,incl_stats=True):
    odir=os.path.join(PSGS_DIR,'psgs_'+fname)
    if not os.path.exists(odir): os.makedirs(odir)
    units = []
    all_units=[]
    done=0
    for i,row in tqdm(df.iterrows(),total=len(df)):
        psg=row.passage.replace('\\\\','\n')
        while '\n\n' in psg: psg=psg.replace('\n\n','\n')
        psg=psg.strip().replace('\n','\n>\t')#.replace("`","'")
        perc_abs=row['num_abs']/row['num_total']*100
        perc_conc=row['num_conc']/row['num_total']*100
        perc_neither=row['num_neither']/row['num_total']*100
        abs_conc = row['num_abs']-row['num_conc']
        unit = f"""



> {psg}
"""
        if incl_stats:
            unit = """

#### Slice #{i+1}

""" + unit
        unit2="""
>    - Abstract words ({row['num_abs']})
>        - {row['abs']}
>    - Concrete words ({row['num_conc']})
>        - {row['conc']}
>    - Neither ({row['num_neither']})
>        - {row['neither']}
>    - Abs - Conc = {abs_conc}
>    - Type / Token = {int(round(row['num_types'] / row['num_tokens'] * 100,0))}


<hr/>
"""

        units.append(unit+unit2 if incl_stats else unit)
        all_units.append((abs_conc,unit))
        if len(units)>=stacklen or i==(len(df)-1):
            done+=1
            ofn = os.path.join(odir,f'psgs_{fname}_{str(done).zfill(4)}.md')
            with open(ofn,'w') as of:
                of.write('\n\n'.join(units))
            units=[]
    
    # save top bottoms
    most_abs = [y for x,y in sorted(all_units,reverse=True)][:stacklen]
    most_conc = [y for x,y in sorted(all_units,reverse=False)][:stacklen]
    most_zero = [y for x,y in sorted(all_units,key=lambda x: abs(x[0]))][:stacklen]
    random.shuffle(all_units)
    most_random = [y for x,y in all_units][:stacklen] #random.sample(all_units,stacklen if stacklen>len(all_units) else len(all_units))]
    ofn_abs = os.path.join(odir,f'most_abs_{fname}.md')
    ofn_conc = os.path.join(odir,f'most_conc_{fname}.md')
    ofn_zero = os.path.join(odir,f'most_zero_{fname}.md')
    ofn_random = os.path.join(odir,f'most_random_{fname}.md')
    with open(ofn_abs,'w') as of: of.write('\n\n'.join(most_abs))
    with open(ofn_conc,'w') as of: of.write('\n\n'.join(most_conc))
    with open(ofn_zero,'w') as of: of.write('\n\n'.join(most_zero))
    with open(ofn_random,'w') as of: of.write('\n\n'.join(most_random))


def to_html(df):
    units = []
    for i,row in df.iterrows():
        unit = f"""

{row.passage}

<br/><br/>
&nbsp;&nbsp;<small>[{row['num_abs']} - {row['num_conc']} = {row['abs-conc']}]</small>

<hr/>
"""
        units.append(unit)
    return '\n\n'.join(units)
