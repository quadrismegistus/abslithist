from abslithist import *

QNUM=1000
ZCUT=.666
USE_COLOR=False

VERSION='v3'






def nodblspc(x):
    while '  ' in x: x=x.replace('  ',' ')
    return x

def corpora_meta(corpora,incl_meta=[]):
    o=[]
    for cname in tqdm(corpora,desc='Loading metadata'):
        cdf=lltk.load(cname).meta
        cdf['corpus']=cname
        o.extend(cdf.reset_index().to_dict('records'))    
    odf=pd.DataFrame(o).set_index(['corpus','id']).sort_index()
    for mk in incl_meta:
        if not mk in odf.columns:
            odf[mk]=''
    odf=odf[incl_meta] if incl_meta else odf
    if 'author' in incl_meta:
        odf['author_lname']=odf.author.apply(lambda x: x.split(',')[0].strip())
    return odf

# corpora_meta(['CanonFiction','MarkMark'])


def get_passages_text(id,corpus=DEFAULT_CORPUS,nmin=50):
    ifn=os.path.join(PATH_SCORES_BYTEXT,corpus,id,'passages.pkl')
    if not os.path.exists(ifn): return pd.DataFrame()
    df = pd.read_pickle(ifn).reset_index()
    if nmin: df=df[df.num_recog>=nmin]
    if 'html' in df.columns: df=df.drop('html',1)
    df['id']=id
    df['corpus']=corpus
    df['txt'] = df.txt.apply(lambda x: x.replace('  ',' '))
    return df

def get_current_psg_scores(
        cachefn=PATH_PSG_CURRENT,
        corpora=['CanonFiction'],
        num_proc=4,
        force=False,
        incl_meta=['year','author','canon_genre','major_genre','subcorpus'],
        **attrs):
    df=None
    if force or not os.path.exists(cachefn):
        objs=[(t.id,cname) for cname in corpora for t in lltk.load(cname).texts()]
        res = pmap(_do_get_current_psg_scores, objs, num_proc=num_proc, **attrs)
        if res is not None and len(res):
            df=pd.concat([df.reset_index() for df in res])
            df=df.sort_values('val')
            df['txt']=df.txt.progress_apply(nodblspc)
            df.to_pickle(cachefn)
    elif os.path.exists(cachefn):
        df=pd.read_pickle(cachefn)
    if df is None or not len(df): return df

    # join with meta?
    dfmeta=corpora_meta(corpora,incl_meta=incl_meta)
    odf=df.set_index(['corpus','id']).join(dfmeta).reset_index()
    odf=odf[~odf.val.isna()].sort_values('val')
    if 'val_perc' in odf.columns:
        odf['val_perc_int']=odf.val_perc.apply(int)
    return odf

def _do_get_current_psg_scores(inp):
    return get_passages_text(*inp)





# # Test
# sentdf=to_sents(willoughby_full)
# sentdf



# # Test
# psgdf=to_psgdf(sentdf)
# psgdf


## Text scores to passages
def divide_by_sent_windows(df,nmin=DEFAULT_NUM_WORDS_IN_PSG):
    o=[]
    ipsg=0
    nnow=0
    df=df.sort_values(['i_para','i_sent'])
    for (para_i,sent_i),dfg in sorted(df.groupby(['i_para','i_sent'])):
        o+=[ipsg for n in range(len(dfg))]
        nnow+=dfg.is_recog.sum()        
        if nnow>=nmin:
            ipsg+=1
            nnow=0
    return o


def do_detokenize_scores(psgdf,valcols=['val','val_perc']):
    psgdf=psgdf.sort_values('i_tok')
#     txt=''.join(psgdf.tok)
    txt=' '.join(
        #' '.join(sentdf.tok)
        detokenize(sentdf.tok)
        for i,sentdf in sorted(psgdf.groupby('i_sent'))
    )
    data=odx={'txt':txt}#, 'html':to_passage_html(psgdf)}
    words=[x for x in psgdf[psgdf.is_punct==0].tokl]
    recogwords=[x for x in psgdf[psgdf.is_recog==1].tokl]
    
    data['i_tok']=min(psgdf.index)
    for icol in ['i_para','i_sent','i_word']:
        data[icol]=min(psgdf[icol])

    data['num_sent']=psgdf.i_sent.nunique()
    data['num_word']=len(words)
    data['num_word_types']=len(set(words))
    data['ttr']=data['num_word_types'] / data['num_word'] if data['num_word'] else np.nan

    data['num_recog']=len(recogwords)
    data['num_recog_types']=len(set(recogwords))
    data['ttr_recog']=data['num_recog_types'] / data['num_recog'] if data['num_recog'] else np.nan


    for vc in valcols:    
        data[vc]=psgdf[vc].mean()
        # data[f'{vc}_mean']=psgdf[vc].mean()
        # data[f'{vc}_median']=psgdf[vc].median()
        # data[f'{vc}_stdev']=psgdf[vc].std()

    return pd.DataFrame([odx])

def to_passages(txt_or_score_df,nmin=DEFAULT_NUM_WORDS_IN_PSG,num_proc=1,progress=False):
    assert type(txt_or_score_df) in {str,pd.DataFrame}
    df=to_scores(txt_or_score_df,sep_para=None) if type(txt_or_score_df)==str else txt_or_score_df
    df['i_psg']=divide_by_sent_windows(df,nmin=nmin)
    return pmap_groups(
        do_detokenize_scores,
        df.groupby('i_psg'),
        num_proc=num_proc,
        progress=progress,
        use_cache=False,
        desc='Parsing passages'
    )






def to_passage_html(txt_or_score_df,valcol='val_perc',show=False):
    assert type(txt_or_score_df) in {str,pd.DataFrame}
    df=to_scores(txt_or_score_df) if type(txt_or_score_df)==str else txt_or_score_df
    
    words=[]
    for i,row in df.fillna('').iterrows():
        word=tok=cleantxt(row.tok)
        if not tok: continue
        if tok=='@':
            words+=[' \n ']
            continue
        if not tok[0].isalpha() and words and tok!='``':
            words[-1]+=tok
            continue
        if row[valcol]:
            val=row[valcol]
            if not isnan(val):
                val_perc_str=str(int(val* 10)).zfill(3)
                word=f'<conc{val_perc_str}>{tok}</conc{val_perc_str}>'
                word=f'<abs>{word}</abs>' if val<50 else f'<conc>{word}</conc>'
        words.append(word)
    xml=f'<p>{" ".join(words)}</p>'
    while '  ' in xml: xml=xml.replace('  ',' ')
    if show:
        init_css()
        printm(xml)
    else:
        return xml

def to_psg_density(
        df,
        other_df=None,
        title='',
        figure_size=(6,2),
        dpi=150,
        valcol='val',
        font_size=6,
        num_runs=100,
        nmin_wiggle=5,
        sample_size=100,
        all_scores=None,
        **attrs):
    p9.options.figure_size=figure_size
    p9.options.dpi=dpi
    
    psgmean=df[valcol].mean()
    avgstr=f'Passage concreteness score average = {round(psgmean,2)}'
    if all_scores is not None:
        from scipy.stats import percentileofscore
        perc=int(round(percentileofscore(all_scores,psgmean)))
        if perc==0: perc=1
        if perc==100: perc=99
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
        avgstr+=f' ({ordinal(perc)} percentile)'
    avgstr2=''
    
    fig=p9.ggplot(p9.aes(x=valcol, y='..density..'))
    dfdat=df.dropna()
    nx=len(df.dropna())
    if other_df is not None and valcol in set(other_df.columns):
        #other_df['i_psg']=divide_by_sent_windows(other_df,nmin=nx)
        #ipsgs=list(set(other_df.i_psg))
        other_dfdat=other_df.dropna()
        for nrun in range(num_runs):
            #smplnum=random.choice(ipsgs)
            #otdfsmpl=other_df.query(f'i_psg=={smplnum}').dropna()
            otdfsmpl=other_dfdat.sample(n=nx,replace=True)
#             lenrun=len(otdfsmpl)
#             if lenrun<nx: continue
#             otdfsmpl=otdfsmpl.iloc[:nx]
            fig+=p9.geom_density(color='silver',data=otdfsmpl,alpha=0.25)
            avgstr2=f'\nText average = {round(other_df[valcol].mean(),2)}'
    fig+=p9.geom_density(data=df)
    if not title: title=f'Distribution of word concreteness scores (n={len(df.dropna())})\n{avgstr}{avgstr2}'
    fig+=p9.labs(
        title=title,
        x='Concreteness score',
        y='Frequency'
    )
    fig+=p9.theme_classic()
    fig+=p9.theme(title=p9.element_text(size=font_size),text=p9.element_text(size=font_size))
    fig+=p9.xlim(-2.25,2.25)
    fig+=p9.geom_vline(xintercept=0,alpha=0.25)
    fig+=p9.geom_text(x=df[valcol].mean()+0.05,y=1,label=avgstr,inherit_aes=False,size=7,ha='left')
    return fig


def showpsg_t(txt,t,title='',charname='',**attrs):
    if not title:
        title=charname+', ' if charname else ''
        title+=f'from {t.au}, _{t.shorttitle}_ ({t.year})'

    return showpsg(
        txt,
        title,
        other_txt=t.txt,
        periods={to_field_period(t.year)},
        **attrs
    )







def tknz(txt):
    p = re.compile(r"\d+|[-'a-z]+|[ ]+|\s+|[.,]+|\S+", re.I)
    slice_starts = [m.start() for m in p.finditer(txt)] + [None]
    return [txt[s:e] for s, e in zip(slice_starts, slice_starts[1:])]

def cleantxt(txt):
#     return txt#.replace("*","").replace('`','').replace("'","").replace('"','').replace('\n',' @ ').replace('\\','')
    return txt.replace('\\n','\n').replace('\\t','\t').replace('\n',' @ ').replace('\\','').replace('*','').replace("`","").replace('_',' ')




def load_scores_text(t):
    try:
        ifn=os.path.join(PATH_SCORES_BYTEXT, t.corpus.name, t.id, 'scores.pkl')
        return pd.read_pickle(ifn)
    except FileNotFoundError:
        pass
    return None




def show(*x,**y): return printimg(showpsg(*x,**y))
def showcmp(*x,**y): return printimg(showpsgs(*x,**y))
def mod(x): return modernize_spelling_in_txt(x, get_spelling_modernizer())


def showpsg(
        txt,
        t=None,
        title='',
        other_txt='',
        showxml=True,
        source='Median',
        stopwords={},
        period=None,
        show=True,
        qnum=QNUM,
        figure_size=(6,4),
        use_color=USE_COLOR,
        dpi=600,
        width=800,
        font_size=6,
        num_runs=30,
        scores_txt=None,
        incl_distro=True,
        show_html=False,
        show_img=True,
        ofn=None):

    # get opts:
    if not stopwords: stopwords=get_stopwords()
    if t is not None:
        if scores_txt is None: scores_txt = load_scores_text(t)
        if not period: period=to_field_period(t.year)
        if not title: title=f'{t.au}, <i>{t.shorttitle}</i> ({t.year})'
        if not ofn: ofn=f'{t.au}.{t.ti}.t{timestamp()}.png'
    else:
        if not period: period='median'
        if scores_txt is None and other_txt:
            scores_txt = to_scores(
                other_txt,
                period=period,
                stopwords=stopwords,
                source=source
            )
    
    # get scores
    from ftfy import fix_text
    scores_psg = to_scores(fix_text(txt),period=period,stopwords=stopwords,source=source)        
    # get passage xml
    xml=to_passage_html(scores_psg,show=False)
    if title: xml=f'<p><b>{title}</b></p>\n{xml}'
    # alt?
    density_fig = to_psg_density(scores_psg, scores_txt, font_size=font_size) if incl_distro else None
    
    
    if show_img:
        opath=showpsgs_img(xmls=[xml],figs=[density_fig],ofn=ofn,width=width)
        return opath
        
    elif show_html:
        if title: printm('#### '+title)#xml='#### '+title+'\n'+xml
        if xml: printm(xml)
        init_css(use_color=use_color)
        if incl_distro: display(density_fig)
        return
    else:
        return (xml,density_fig)
    
def showpsg_html(*x,**y):
    y['show_html'],y['show_img']=True,False
    return showpsg(*x,**y)
    



def showpsgs_img(xmls,figs=[],ofn=None,width=800):
    fightms=[]
    for fi,fig in enumerate(figs):
        tmpfigfn=os.path.abspath(f'.fig.density.{fi}.png')
        fig.save(tmpfigfn)
        fightm=f'<img src="{tmpfigfn}" width="{width//len(figs)}" />'
        fightms.append(fightm)
        
    extra_css="""
    <style type="text/css">
    td { width: """+str(width//len(xmls))+"""px; line-height:2em; }
    tr,td,th,table { border:1px solid gray; padding:0.5em; }
    th { text-align: center; font-weight:normal; font-size:1.1em; } 
    tr { vertical-align: top; }
    table {
      border-collapse: collapse;
    }
    </style>
    """
    
    html_str = f"""
    <html>
    <head>
    <title>Comparison</title>
    {get_css()}
    {extra_css}
    </head>
    <body width="{width}">
    <center>
    <table width="{width}">
    <tr>
    <td>
    """ + '</td><td>'.join([
        xmls[i] + ('<br/>'+fightms[i] if len(fightms)>i else '')
        for i in range(len(xmls))
    ]) + """
    </td>
    </tr>
    </table>
    </center>
    </body>
    </html>
    """

    from ftfy import fix_text
    html_str=fix_text(lltk.clean_text(html_str))

    tmphtmfn=os.path.abspath('.fig.htm.html')
    if not ofn: ofn='.fig.now.png'
    if not os.path.isabs(ofn): ofn=os.path.join(PATH_FIGS,ofn)
    odir=os.path.dirname(ofn)
    if odir and not os.path.exists(odir): os.makedirs(odir)
    with open(tmphtmfn,'w') as of: of.write(html_str)
    abscmd=os.path.join(PATH_HERE,'')
    cmd=f'{PATH_IMGCONVERT} "{tmphtmfn}" "{ofn}"'
    x=os.system(cmd)
    return os.path.abspath(ofn)
    
    
    
    
    
    
    
    
    
def showpsgs(argsets,ofn=None,odir=PATH_FIGS,**kwargs):
    xml_figs = [showpsg(*args,show_img=False,show_html=False,**kwargs) for args in argsets]
    xmls,figs=zip(*xml_figs)
    if not ofn:
        ts=[x[1]
            for x in argsets
           if len(x)>1]
        akey='-'.join([t.ti for t in ts])
        ofn=f'fig.compare.{akey+"." if akey else ""}t{timestamp()}.png'
    opath=showpsgs_img(xmls,figs,ofn=ofn)
    return opath

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def compare_psgs_table(inpdf,width=555,ofn='fig.psg_tbl.png',**attrs):
    inpdf['year']=inpdf.t.apply(lambda t: t.year)
    inpdf['txt']=inpdf.txt.apply(lambda x: x.strip())
    inpdf['xml']=inpdf.txt.apply(lambda x: showpsg(x,show=False,incl_distro=False,**attrs)[0])
    if not 'title' in inpdf: inpdf['title']=inpdf.t.apply(lambda t: t.ti)
    inpdf=inpdf.sort_values('year')
    
    dfx=inpdf.pivot('label','title','xml')
    dfx=dfx[sorted(dfx.columns, key=lambda col: inpdf.query(f'title=="{col}"').year.mean())]
    
    dfhtml=to_simple_html(dfx)
    
    # o html
    extra_css="""
    <style type="text/css">
    td { width: """+str(width)+"""px; line-height:2em; padding: 1em; border:1px solid gray; }
    tr,td,th,table { border:1px solid gray; }
    th { text-align: center; font-weight:normal; font-size:1.1em; } 
    tr { vertical-align: top; }
    </style>
    """
    html_str = f"""
    <html>
    <head>
    <title>Comparison</title>
    {get_css()}
    {extra_css}
    </head>
    <body>
    {dfhtml}
    </body>
    </html>
    """
    from ftfy import fix_text
    html_str=fix_text(lltk.clean_text(html_str))

    return htm2png(html_str,ofn)

# def compare_psgs(
#         txts=[],
#         ts=[],
#         charnames=[],
#         titles=[],
#         width=444,
#         ofn=None,
#         show=False,
#         font_size=9,
#         monospace=False,
#         **attrs):
#     md1,fig1=showpsg_t(txts[0],t=ts[0],charname=charnames[0],show=show,font_size=font_size,**attrs)
#     md2,fig2=showpsg_t(txts[1],t=ts[1],charname=charnames[1],show=show,font_size=font_size,**attrs)    
#     tmpfn1='.fig.1.png'
#     tmpfn2='.fig.2.png'
#     fig1.save(tmpfn1)
#     fig2.save(tmpfn2)
    
#     dfx=pd.DataFrame([
#         {'passage':md1,'figure':f'<img src="{tmpfn1}" width="{width}" />'},
#         {'passage':md2,'figure':f'<img src="{tmpfn2}" width="{width}" />'},
#     ]).T.reset_index().drop('index',1)
    
#     dfx.columns = titles if titles else [
#         (charname+', ' if charname else '')+ f'from {t.au}, <i>{t.shorttitle}</i> ({t.year})'
#         for charname,t in zip(charnames,ts)
#     ]
#     from ftfy import fix_text
        
    
#     dfhtml=dfx.to_html(index=False).replace('&gt;','>').replace('&lt;','<').replace('\\n','').replace('  ',' ')    
    

#     extra_css="""
#     <style type="text/css">
#     td { width: """+str(width)+"""px; line-height:2em; }
#     tr,td,th,table { border:0px; }
#     th { text-align: center; font-weight:normal; font-size:1.1em; } 
#     tr { vertical-align: top; }
#     """

#     if monospace:
#         extra_css+="""
#         table { font-size:0.8em; font-family: Menlo,"Courier New",monospace; }
#         """
#     extra_css+="""
#     </style>
#     """
    
#     html_str = f"""
#     <html>
#     <head>
#     <title>Comparison</title>
#     {get_css()}
#     {extra_css}
#     </head>
#     <body>
#     {dfhtml}
#     </body>
#     </html>
#     """

#     html_str=fix_text(lltk.clean_text(html_str))



#     tmphtmfn=os.path.abspath('.fig.htm.html')
#     if not ofn: ofn=os.path.join(PATH_FIGS,'psgcompare',VERSION,f'{charnames[0]}-v-{charnames[1]}.png')
#     if not os.path.exists(os.path.dirname(ofn)): os.makedirs(os.path.dirname(ofn))
#     with open(tmphtmfn,'w') as of: of.write(html_str)
#     abscmd=os.path.join(PATH_HERE,'')
#     # print(PATH_IMGCONVERT,os.path.exists(PATH_IMGCONVERT))
#     # print(tmphtmfn,os.path.exists(tmphtmfn))
#     # print(ofn,os.path.exists(ofn))
#     cmd=f'{PATH_IMGCONVERT} "{tmphtmfn}" "{ofn}"'
#     # print('>>',cmd)
#     x=os.system(cmd)
#     # print('<<',x)
#     if PATH_FIGS2: os.system(f'cp {ofn} {PATH_FIGS2}')
#     return os.path.abspath(ofn)

# def compare_psgs_show(*x,**y):
#     printimg(compare_psgs(*x,**y))
















def get_css(use_color=USE_COLOR,qnum=QNUM,zf=3):
    zf=3
    labels=[str(x).zfill(zf) for x in range(qnum)]
    from colour import Color


    # Set colors
    if use_color:
        gradient=rdybl=list(reversed([
        #     '#A51626',
            '#D73027',
            '#F46D43',
            '#FDAE61',
            '#FEE090',
            '#FFFDBF',
            '#E0F3F8',
            '#ABD9E9',
            '#74ADD1',
            '#4575B4',
        #     '#323695'
        ]))
        len(rdybl)
    else:
        gradient=['#EFEFEF','#9A9A9A','#EFEFEF']
    # gradient=['silver','white','silver']

    color_range=[]
    for i,x in enumerate(gradient):
        if not i: continue
        prev=gradient[i-1]
        color_range+=list(
            Color(prev).range_to(
                Color(x),
                int(round(qnum/(len(gradient)-1)))
            )
        )
    if use_color:
        for c in color_range:
            c.saturation=0.6
            c.luminance = 0.75
    width_range=4
    bold_range=1000
    lastpoint=len(color_range)
    midpoint=len(color_range)//2
    opacity_range=0.2
    
    css = []
    css+=['abs,conc,neither { border: '+str(width_range//2)+'px; height:1.5em; display: inline-block }']
    css+=['abs {font-style:italic; }']

    for i,x in enumerate(color_range):
        diff=abs(midpoint-i)
        diffperc=int(round(diff/midpoint * width_range))
        fweight=int(round(diff/midpoint * bold_range)) + 1

        opacity= (diff/midpoint *opacity_range) + 0.025
        cssx=f'conc{str(i).zfill(zf)}' + ' {'
        rgbstr=str(x.rgb)[:-1] + f', {opacity})'
        if i>=midpoint:
            cssx+=f'background-color: rgba{rgbstr}; '
            cssx+=f'font-weight: {fweight}; '
        else:
            cssx+=f'border: {diffperc}px solid dimgray; '
        cssx+=' }'
        css+=[cssx]
    css+=['p { line-height: 1.5em; }']
    cssstr=f'<style type="text/css">' + ("\n".join(css)) + '</style>'
    return cssstr


def init_css(use_color=USE_COLOR):
    printm(get_css(use_color=use_color))


def path2psgs(path,*x,**y):
    with open(path) as f: txt=f.read()
    return pd.concat(txt2psgs(txt,*x,**y))


# Collapse into single row
def maxint(x): return int(max(x))
def nuniq(x): return len(set(x))
def nword(toks):
    return len([x for x in toks if type(x)==str and x and x[0].isalpha()])
    


### From realism.py --> still nec?

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


####







#####
# CANONICAL
#####

darcy_full="""
His friend Mr. Darcy soon drew the attention of the room by his fine, tall 
person, handsome features, noble mien; and the report which was in general 
circulation within five minutes after his entrance, of his having ten thousand a 
year. The gentlemen pronounced him to be a fine figure of a man, the ladies 
declared he was much handsomer than Mr. Bingley, and he was looked at with great 
admiration for about half the evening, till his manners gave a disgust which 
turned the tide of his popularity; for he was discovered to be proud, to be 
above his company, and above being pleased; and not all his large estate in 
Derbyshire could then save him from having a most forbidding, disagreeable 
countenance, and being unworthy to be compared with his friend.
"""

willoughby_full="""
A gentleman carrying a gun, with two pointers playing round him, was passing 
up the hill and within a few yards of Marianne, when her accident happened. He 
put down his gun and ran to her assistance. She had raised herself from the 
ground, but her foot had been twisted in her fall, and she was scarcely able to 
stand. The gentleman offered his services; and perceiving that her modesty 
declined what her situation rendered necessary, took her up in his arms without 
farther delay, and carried her down the hill. Then passing through the garden, 
the gate of which had been left open by Margaret, he bore her directly into the 
house, whither Margaret was just arrived, and quitted not his hold till he had 
seated her in a chair in the parlour.

Elinor and her mother rose up in amazement at their entrance, and while the 
eyes of both were fixed on him with an evident wonder and a secret admiration 
which equally sprung from his appearance, he apologized for his intrusion by 
relating its cause, in a manner so frank and so graceful that his person, which 
was uncommonly handsome, received additional charms from his voice and 
expression. Had he been even old, ugly, and vulgar, the gratitude and kindness 
of Mrs. Dashwood would have been secured by any act of attention to her child; 
but the influence of youth, beauty, and elegance, gave an interest to the action 
which came home to her feelings.

She thanked him again and again; and, with a sweetness of address which always 
attended her, invited him to be seated. But this he declined, as he was dirty 
and wet. Mrs. Dashwood then begged to know to whom she was obliged. His name, he 
replied, was Willoughby, and his present home was at Allenham, from whence he 
hoped she would allow him the honour of calling tomorrow to enquire after Miss 
Dashwood. The honour was readily granted, and he then departed, to make himself 
still more interesting, in the midst of a heavy rain.

His manly beauty and more than common gracefulness were instantly the theme of 
general admiration, and the laugh which his gallantry raised against Marianne 
received particular spirit from his exterior attractions. Marianne herself had 
seen less of his person than the rest, for the confusion which crimsoned over 
her face, on his lifting her up, had robbed her of the power of regarding him 
after their entering the house. But she had seen enough of him to join in all 
the admiration of the others, and with an energy which always adorned her 
praise. His person and air were equal to what her fancy had ever drawn for the 
hero of a favourite story; and in his carrying her into the house with so little 
previous formality, there was a rapidity of thought which particularly 
recommended the action to her. Every circumstance belonging to him was 
interesting. His name was good, his residence was in their favourite village, 
and she soon found out that of all manly dresses a shooting-jacket was the most 
becoming. Her imagination was busy, her reflections were pleasant, and the pain 
of a sprained ankle was disregarded.
"""

marianne_full="""
Marianne's abilities were, in many respects, quite equal to Elinor's. She was 
sensible and clever; but eager in everything: her sorrows, her joys, could have 
no moderation. She was generous, amiable, interesting: she was everything but 
prudent. The resemblance between her and her mother was strikingly great.

Elinor saw, with concern, the excess of her sister's sensibility; but by Mrs. 
Dashwood it was valued and cherished. They encouraged each other now in the 
violence of their affliction. The agony of grief which overpowered them at 
first, was voluntarily renewed, was sought for, was created again and again. 
They gave themselves up wholly to their sorrow, seeking increase of wretchedness 
in every reflection that could afford it, and resolved against ever admitting 
consolation in future. Elinor, too, was deeply afflicted; but still she could 
struggle, she could exert herself. She could consult with her brother, could 
receive her sister-in-law on her arrival, and treat her with proper attention; 
and could strive to rouse h
"""




ferrars_full="""Edward Ferrars was not recommended to their good opinion by any peculiar graces 
of person or address. He was not handsome, and his manners required intimacy to 
make them pleasing. He was too diffident to do justice to himself; but when his 
natural shyness was overcome, his behaviour gave every indication of an open, 
affectionate heart. His understanding was good, and his education had given it 
solid improvement. But he was neither fitted by abilities nor disposition to 
answer the wishes of his mother and sister, who longed to see him 
distinguished-- as-- they hardly knew what. They wanted him to make a fine 
figure in the world in some manner or other. His mother wished to interest him 
in political concerns, to get him into parliament, or to see him connected with 
some of the great men of the day. Mrs. John Dashwood wished it likewise; but in 
the mean while, till one of these superior blessings could be attained, it would 
have quieted her ambition to see him driving a barouche. But Edward had no turn 
for great men or barouches. All his wishes centered in domestic comfort and the 
quiet of private life."""