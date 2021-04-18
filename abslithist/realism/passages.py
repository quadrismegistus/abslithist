import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..'))
from abslithist import *

QNUM=1000
ZCUT=.666
USE_COLOR=False

VERSION='v3'

def get_current_psg_scores():
    dfpsg = pd.read_pickle(PATH_PSG_SCORE).reset_index()
    return dfpsg



# Parse raw text into paras/sentences

def to_sentdf(txt,words_recog=set(),num_word_min=45,vald={},valname='val',sep_para='\n\n'):
    ntok,nword,npara,nsent=0,-1,0,0
    txt=txt.strip()
    o=[]
    
    def _cleantok(x):
        return {
            ' ':'_',
            '\n':'|'
        }.get(x,x)
    
    for pi,para in enumerate(txt.split(sep_para)):
        para=para.strip()
        for si,sent in enumerate(tokenize_sentences(para)):
            sent=sent.strip()
            swords=tokenize_agnostic(sent)
            swords=[_cleantok(x) for x in swords]
            for w in swords:
                w=w.strip()
                wl=w.lower()
                ispunc=int(not w or not w[0].isalpha())
                if not ispunc: nword+=1
                dx={
                    'i_para':npara,
                    'i_sent':nsent,
                    'i_word':nword,# if not ispunc else None,
                    'i_tok':ntok,
                    'tok':w,
                    'tokl':wl,
                    'is_punct':ispunc,
                }
                if words_recog: dx['is_recog']=int((w in words_recog) or (wl in words_recog))
                if vald: dx[valname]=vald.get(wl,np.nan)
                o+=[dx]
                ntok+=1
            nsent+=1
        npara+=1
    odf=pd.DataFrame(o)
    return odf.set_index('i_tok') if 'i_tok' in odf.columns else odf

# # Test
# sentdf=to_sentdf(willoughby_full)
# sentdf

def to_psgdf(sentdf_or_txt,tfield='Abs-Conc.Median.median',norms=None,tokname='tokl',valname='val',nmin=50,stopwords=set()):
    
    if norms is None: norms=get_allnorms()
    normsok=norms[tfield].dropna()
    stopwords|=get_stopwords()
    wordsok=set(normsok.index) - stopwords
    w2score=dict((a,b) for a,b in zip(normsok.index, normsok) if a not in stopwords)
    sentdf = to_sentdf(sentdf_or_txt) if type(sentdf_or_txt)==str else sentdf_or_txt
    if tokname in sentdf.columns:
        sentdf['is_recog']=sentdf[tokname].apply(lambda w: int(w in wordsok))
        sentdf[valname]=sentdf[tokname].apply(lambda w: w2score.get(w))
    if valname in sentdf.columns:
        scores=pd.Series(w2score.values())
        sentdf[valname+'_perc']=sentdf[valname].apply(lambda x: percentileofscore(scores,x))
    return sentdf#.set_index('i_tok')

# # Test
# psgdf=to_psgdf(sentdf)
# psgdf


def to_psg_html(df,valcol='val_perc'):
    words=[]
    for i,row in df.fillna('').iterrows():
        word=tok=cleantxt(row.tok)
        if tok=='@':
            words+=[' \n ']
            continue
        if not tok[0].isalpha() and words and tok!='``':
            words[-1]+=tok
            continue
        if row[valcol]:
            val=row[valcol]
            if not np.isnan(val):
                val_perc_str=str(int(val* 10)).zfill(3)
                word=f'<conc{val_perc_str}>{tok}</conc{val_perc_str}>'
                word=f'<abs>{word}</abs>' if val<50 else f'<conc>{word}</conc>'
        words.append(word)
    xml=f'<p>{" ".join(words)}</p>'
    while '  ' in xml: xml=xml.replace('  ',' ')
    return xml

def to_psg_density(
        df,
        other_txt='',
        title='',
        figure_size=(6,2),
        dpi=600,
        valcol='val',
        font_size=6,
        num_runs=30,
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
    
    fig=p9.ggplot(p9.aes(x=valcol))
    if other_txt:
        nx=len(df[valcol].dropna())
        otdf=to_psgdf(other_txt,stopwords=stopwords)
        otdf=otdf[~otdf[valcol].isna()]
        otdfavgs=[]
        for nn in range(num_runs):
            otdfs=otdf.sample(n=nx,replace=True)
            fig+=p9.geom_density(color='silver',data=otdfs,alpha=0.25)
            otdfavgs.append(otdfs[valcol].mean())
        avgstr2=f'\nText average = {round(otdf[valcol].mean(),2)}'
    fig+=p9.geom_density(data=df)
    if not title: title=f'Distribution of word concreteness scores (n={len(df.dropna())})\n{avgstr}{avgstr2}'
    fig+=p9.labs(
        title=title,
        x='Concreteness score',
        y='Frequency'
    )
    fig+=p9.theme_classic()
    fig+=p9.theme(title=p9.element_text(size=font_size),text=p9.element_text(size=font_size))
    fig+=p9.xlim(-3,3)
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







def df2xml(df,abs_below=-ZCUT,conc_above=ZCUT,qnum=1000,zf=3,valcol='z'):
    """ @DEPRECATED """

    words=[]
    if not 'bin' in df:
        labels=[
            str(x).zfill(zf) if not np.isnan(x) else ''
            for x in range(qnum)
        ]
        df['bin']=[str(x) for x in pd.qcut(df[valcol],q=qnum,labels=labels,duplicates='drop')]
    
    for i,row in df.fillna('').iterrows():
        tok=cleantxt(row.tok)
        if tok=='@':
            words+=[' \n ']
            continue
        # if not tok: continue
        if not tok[0].isalpha() and words and tok!='``':
            words[-1]+=tok
            continue
        if row.bin:
            word=f'<conc{row.bin}>{tok}</conc{row.bin}>'
        else:
            word=tok
        # ?
        if row[valcol]:
            z=float(row[valcol])
            if z<abs_below:
                word=f'<abs>{word}</abs>'
            elif z>conc_above:
                word=f'<conc>{word}</conc>'
            else:
                word=f'<neither>{word}</neither>'            
        words.append(word)
    # xml='<p>' + ' '.join(words) + '</p>'
    # xml=f'<p>{cleanstrip(detokenize(words))}</p>'
    xml=f'<p>{" ".join(words)}</p>'
    # xml=xml.replace(' ,',',')
    return xml



def plot_densityz(df,title='',fig=None):
    p9.options.dpi=600
    p9.options.figure_size=(8,4)
    df=df.dropna()

    # density plot
    avgstr=f'Average = {round(df.z.mean(),2)}'
    
    if fig is None:
        fig=p9.ggplot(p9.aes(x='z',y='..count..'))
    for nn in range(25):
        fig+=p9.geom_density(data=df.sample(n=1000,replace=True))

    return fig

PSG_NORMS=None
def get_psg_norms(cachefn='.data.cache.dfnorms.pkl'):
    global PSG_NORMS
    if PSG_NORMS is None:
        import lltk
        if os.path.exists(cachefn):
            # print('>> reading from:',cachefn)
            PSG_NORMS=lltk.read_df(cachefn)
        else:
            print('>> loading norms...')
            PSG_NORMS = format_norms_as_long(get_allnorms())
            print('>> saving to:',cachefn)
            lltk.save_df(PSG_NORMS,cachefn)
    return PSG_NORMS

def tknz(txt):
    p = re.compile(r"\d+|[-'a-z]+|[ ]+|\s+|[.,]+|\S+", re.I)
    slice_starts = [m.start() for m in p.finditer(txt)] + [None]
    return [txt[s:e] for s, e in zip(slice_starts, slice_starts[1:])]

def psg2df(txt,stopwords={},periods={'median'},dfnorms=None,qnum=QNUM,zf=3,sources={'Median'}):
    labels=[str(x).zfill(zf) for x in range(qnum)]
    if dfnorms is None: dfnorms=get_psg_norms()
    norms=dfnorms if not periods else dfnorms[dfnorms.period.isin(periods)]
    if sources: norms=norms[norms.source.isin(sources)]
    wordavg=norms[['word','z']].groupby('word').mean()
    wordavg['bin']=[str(x) for x in pd.qcut(wordavg.z,q=qnum,labels=labels)]
    txt=cleantxt(txt)
    txtl=tknz(txt)
    df=pd.DataFrame()
    df['tok']=txtl
    df['i']=list(range(len(txtl)))
    df['tokl']=df['tok'].apply(lambda x: x.lower())
    df['tokl_mod']=df['tokl'].apply(lambda x: get_spelling_modernizer().get(x,x).lower())
    df=df.set_index('tokl_mod').join(wordavg,how='left').rename_axis('tokl_mod').reset_index().sort_values('i')
    # df=df.set_index('tokl').join(wordavg,how='left').rename_axis('tokl').reset_index().sort_values('i')
    #df.loc[df['tokl'].isin(stopwords),'z']=np.nan
    df['z']=[np.nan if tokl in stopwords else x
            for tokl,x in zip(df.tokl, df.z)]
    df['bin']=['' if tokl in stopwords else x
            for tokl,x in zip(df.tokl, df.bin)]
    return df

def cleantxt(txt):
#     return txt#.replace("*","").replace('`','').replace("'","").replace('"','').replace('\n',' @ ').replace('\\','')
    return txt.replace('\\n','\n').replace('\\t','\t').replace('\n',' @ ').replace('\\','').replace('*','').replace("`","").replace('_',' ')

def showpsg(txt,title='',other_txt='',showxml=True,stopwords={},periods={},show=True,qnum=QNUM,figure_size=(6,2),incl_distro=True,use_color=USE_COLOR,dpi=600,font_size=6,num_runs=30):
    df=psg2df(cleantxt(txt),stopwords=stopwords,periods=periods)
    xml=df2xml(df)
    if title and xml and show: printm('#### '+title)#xml='#### '+title+'\n'+xml
    if show and xml: printm(xml)
    avgstr=f'Passage average = {round(df.z.mean(),2)}'
    avgstr2=''
    # nx=1000
    nx=len(df.z.dropna())
    #if nx<10: nx=10
    
    # density plot
    fig=None
    if incl_distro:
        fig=p9.ggplot(p9.aes(x='z'))
        if other_txt:
            otdf=psg2df(cleantxt(other_txt),stopwords=stopwords)
            otdf=otdf[~otdf.z.isna()]
            otdfavgs=[]
            for nn in range(num_runs):
                otdfs=otdf.sample(n=nx,replace=True)
                fig+=p9.geom_density(color='silver',data=otdfs,alpha=0.25)
                otdfavgs.append(otdfs.z.mean())
            # avgstr2=f'\nText average = {round(np.mean(otdfavgs),2)}'
            avgstr2=f'\nText average = {round(otdf.z.mean(),2)}'

        fig+=p9.geom_density(data=df)
        # other densities?
        
        p9.options.figure_size=figure_size
        p9.options.dpi=dpi
        fig+=p9.labs(
            title=f'Distribution of word concreteness scores (n={len(df.dropna())})\n{avgstr}{avgstr2}',
            x='Concreteness score',
            y='Frequency'
        )
        fig+=p9.theme_classic()
        fig+=p9.theme(title=p9.element_text(size=font_size),text=p9.element_text(size=font_size))
        fig+=p9.xlim(-3,3)
    #     fig+=p9.geom_vline(xintercept=df.z.mean(),alpha=0.5)
        fig+=p9.geom_vline(xintercept=0,alpha=0.25)
        fig+=p9.geom_text(x=df.z.mean()+0.05,y=1,label=avgstr,inherit_aes=False,size=7,ha='left')
        if show:
            init_css(use_color=use_color)
            display(fig)
            return
    # otherwise
    return xml,fig
    
#     return df





# def get_current_dfall_psgs(corpus_name='CanonFiction'):
#     cdf=pd.DataFrame(readgen_jsonl(PATH_PSGS))
#     cdf=cdf[cdf.num_total == cdf.num_total.max()]
#     cdf['abs-conc']=cdf['num_abs']-cdf['num_conc']
#     cdf['abs/conc']=cdf['num_abs']/cdf['num_conc']

#     import lltk
#     C=lltk.load(corpus_name)
#     cdf['abs/conc']=cdf['num_abs']/cdf['num_conc']
#     cdf['abs-conc']=cdf['num_abs']-cdf['num_conc']
#     cdf['abs-conc_z']=zscore(cdf['abs-conc'])
#     dfall=cdf.merge(C.metadata,on='id').sort_values('abs-conc')
#     dfall=dfall[
#         ['id','year','title','author','passage','abs-conc','abs-conc_z',
#          'canon_genre','major_genre',
#         'num_abs','num_conc','num_neither','num_types','num_total','num_tokens']
#     ]
#     return dfall


# def compare_extremes(dfall,corpus_name='CanonFiction',topn=100):
#     C=lltk.load(corpus_name)
#     most_conc = dfall.head(topn)
#     most_abs = dfall[dfall.title!='The Making of Americans'].sort_values('abs-conc',ascending=False).head(topn)
    
#     for i_conc,(index_conc,row_conc) in enumerate(tqdm(list(most_conc.iterrows()))):
#         idx_conc=row_conc.id
#         t_conc=C.textd[idx_conc]

#         i_abs=i_conc
#         row_abs=most_abs.iloc[i_conc]
#         idx_abs=row_abs.id
#         t_abs=C.textd[idx_conc]
        
#         fn=compare_psgs(
#             (row_abs.passage, row_conc.passage),
#             (t_abs, t_conc),
#             (f'#{i_abs+1} most abstract passage',f'#{i_conc+1} most concrete passage'),
#         )
#         display(fn)
#         display(Image(filename=fn))
#         break
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
    td { width: """+str(width)+"""px; line-height:2em; }
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

def compare_psgs(
        txts=[],
        ts=[],
        charnames=[],
        titles=[],
        width=444,
        ofn=None,
        show=False,
        font_size=9,
        monospace=False,
        **attrs):
    md1,fig1=showpsg_t(txts[0],t=ts[0],charname=charnames[0],show=show,font_size=font_size,**attrs)
    md2,fig2=showpsg_t(txts[1],t=ts[1],charname=charnames[1],show=show,font_size=font_size,**attrs)    
    tmpfn1='.fig.1.png'
    tmpfn2='.fig.2.png'
    fig1.save(tmpfn1)
    fig2.save(tmpfn2)
    
    dfx=pd.DataFrame([
        {'passage':md1,'figure':f'<img src="{tmpfn1}" width="{width}" />'},
        {'passage':md2,'figure':f'<img src="{tmpfn2}" width="{width}" />'},
    ]).T.reset_index().drop('index',1)
    
    dfx.columns = titles if titles else [
        (charname+', ' if charname else '')+ f'from {t.au}, <i>{t.shorttitle}</i> ({t.year})'
        for charname,t in zip(charnames,ts)
    ]
    from ftfy import fix_text
        
    
    dfhtml=dfx.to_html(index=False).replace('&gt;','>').replace('&lt;','<').replace('\\n','').replace('  ',' ')    
    

    extra_css="""
    <style type="text/css">
    td { width: """+str(width)+"""px; line-height:2em; }
    tr,td,th,table { border:0px; }
    th { text-align: center; font-weight:normal; font-size:1.1em; } 
    tr { vertical-align: top; }
    """

    if monospace:
        extra_css+="""
        table { font-size:0.8em; font-family: Menlo,"Courier New",monospace; }
        """
    extra_css+="""
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

    html_str=fix_text(lltk.clean_text(html_str))



    tmphtmfn=os.path.abspath('.fig.htm.html')
    if not ofn: ofn=os.path.join(PATH_FIGS,'psgcompare',VERSION,f'{charnames[0]}-v-{charnames[1]}.png')
    if not os.path.exists(os.path.dirname(ofn)): os.makedirs(os.path.dirname(ofn))
    with open(tmphtmfn,'w') as of: of.write(html_str)
    abscmd=os.path.join(PATH_HERE,'')
    # print(PATH_IMGCONVERT,os.path.exists(PATH_IMGCONVERT))
    # print(tmphtmfn,os.path.exists(tmphtmfn))
    # print(ofn,os.path.exists(ofn))
    cmd=f'{PATH_IMGCONVERT} "{tmphtmfn}" "{ofn}"'
    # print('>>',cmd)
    x=os.system(cmd)
    # print('<<',x)
    if PATH_FIGS2: os.system(f'cp {ofn} {PATH_FIGS2}')
    return os.path.abspath(ofn)

def compare_psgs_show(*x,**y):
    printimg(compare_psgs(*x,**y))















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

def year2period(y):
    if y<1600: return 'C16'
    if y>1900: return 'C20'
    return f'C{(y//100)+1}'

def init_css(use_color=USE_COLOR):
    printm(get_css(use_color=use_color))
def year2period(y):
    if y<1600: return 'C16'
    if y>1900: return 'C20'
    return f'C{(y//100)+1}'


dfall_cols = ['id','year','title','author','passage','abs-conc','abs-conc_z',
            'num_abs','num_conc','num_neither','num_types','num_total','num_tokens',
            'major_genre','canon_genre']

def get_current_dfall_psgs(corpus_name=DEFAULT_CORPUS,meta_cols=['id','year','title','author','major_genre','canon_genre']):
    df=read_df(PATH_PSG_CURRENT).merge(
        lltk.load(corpus_name).meta.reset_index()[meta_cols].reset_index(),
        on='id',
        how='left'
    )
    return df

# def get_current_dfall_psgs(corpus_name=DEFAULT_CORPUS,cachefn=None):
#     if not cachefn: cachefn=os.path.join(PATH_DATA,'psgs','cache.dfall.psgs.pkl')
#     # os.system(f'rm {cachefn}')
#     if not os.path.exists(cachefn):
#         cdf=pd.DataFrame(readgen_jsonl(PATH_PSGS))
#         cdf=cdf[cdf.num_total == cdf.num_total.max()]
#         cdf['abs-conc']=cdf['num_abs']-cdf['num_conc']
#         cdf['abs/conc']=cdf['num_abs']/cdf['num_conc']

#         C=lltk.load(corpus_name)
#         cdf['abs/conc']=cdf['num_abs']/cdf['num_conc']
#         cdf['abs-conc']=cdf['num_abs']-cdf['num_conc']
#         cdf['abs-conc_z']=zscore(cdf['abs-conc'])
#         dfall=cdf.merge(C.metadata,on='id').sort_values('abs-conc')
#         for c in dfall_cols:
#             if not c in set(dfall.columns):
#                 dfall[c]=''
#         dfall=dfall[dfall_cols]
#         save_df(dfall,cachefn)
#         return dfall
#     else:
#         return read_df(cachefn)


def saveextremes(dfall, name='concrete'):
    topn=25
    most_conc = dfall.head(topn)
    most_abs = dfall[dfall.title!='The Making of Americans'].sort_values('abs-conc',ascending=False).head(topn)

    for i,(index,row) in enumerate(tqdm(list(most.iterrows()))):
        idx=row.id
        t=C.textd[idx]
        title=f'''#{i+1} most {name} passage: {row.author.split(',')[0]}, _{row.title[:50]}_ ({row.year})'''
        showpsg(
            row.passage,
            title,
            t.txt,
            periods={year2period(t.year)}
        )
    #     break




# def txt2psgs1(txt,psg_len=50,bysent=True):
#     sentdfs=[]
#     nw=0
#     si=0
#     pi=0
#     for sent in tqdm(tokenize_sentences(txt)):
#         sentdf=psg2df(sent)
#         sentdf['i_sent']=si
#         si+=1
#         sentdfs.append(sentdf)        
#         sentnw=len(sentdf.z.dropna())
#         nw+=sentnw
#         if nw>=psg_len:
#             odf=pd.concat(sentdfs)
#             odf['i']=list(range(len(odf)))
#             odf['i_psg']=pi
#             pi+=1
#             yield odf
#             sentdfs=[]
#             nw=0
#     if len(sentdfs):
#         odf=pd.concat(sentdfs)
#         odf['i']=list(range(len(odf)))
#         odf['i_psg']=pi
#         yield odf

def txt2psgs(txt,psg_len=50,bysent=True,num_proc=1,**y):
    sentdfs=[]
    nw=0
    si=0
    pi=0
    iterr=pmap_iter(
        psg2df,
        tokenize_sentences(txt),
        desc='Scoring passages',
        num_proc=num_proc
    )
    for si,sentdf in enumerate(iterr):
        sentdf['i_sent']=si
        sentdfs.append(sentdf)        
        sentnw=len(sentdf.z.dropna())
        nw+=sentnw
        if nw>=psg_len:
            odf=pd.concat(sentdfs)
            odf['i']=list(range(len(odf)))
            odf['i_psg']=pi
            pi+=1
            yield odf
            sentdfs=[]
            nw=0
    if len(sentdfs):
        odf=pd.concat(sentdfs)
        odf['i']=list(range(len(odf)))
        odf['i_psg']=pi
        yield odf

def path2psgs(path,*x,**y):
    with open(path) as f: txt=f.read()
    return pd.concat(txt2psgs(txt,*x,**y))


# Collapse into single row
def maxint(x): return int(max(x))
def nuniq(x): return len(set(x))
def nword(toks):
    return len([x for x in toks if type(x)==str and x and x[0].isalpha()])
    
def psgdf2row(psgdf):
    psgdf=psgdf.sort_values('i')
    txt=cleantxt(detokenize(psgdf.tok).replace('@','')).strip()
    xml=df2xml(psgdf)
    while '  ' in txt: txt=txt.replace('  ',' ').replace(' .','.')
    odx={'txt':txt,'xml':xml}
    data=psgdf.agg({
        'i':maxint,
        'z':np.mean,
        'i_sent':maxint,
        'i_psg':maxint
    })
    words=[x for x in psgdf.tokl if x and x[0].isalpha()]
    data['num_row']=len(psgdf)
    data['num_word']=len(words)
    data['num_word_types']=len(set(words))
    data['num_recog']=len(psgdf.dropna().tokl)
    data['num_recog_types']=len(set(psgdf.dropna().tokl))
    data['num_sent']=nuniq(psgdf.i_sent)
    data['num_word_types']=len(set(psgdf.tokl))
    data['ttr_recog']=data['num_recog_types'] / data['num_recog']
    data['ttr']=data['num_word_types'] / data['num_word']
    for k,v in dict(data).items(): odx[k]=v
    return odx


def txt2psgrows(txt):
    return pd.DataFrame(
        psgdf2row(psgdf)
        for psgdf in txt2psgs(txt)
    )


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