#!/usr/bin/env python
# coding: utf-8

# # Abstract novels

# ## Setup

# In[1]:


import lltk,pickle,os,json, numpy as np
import pandas as pd,numpy as np
from collections import defaultdict


# In[2]:


import lltk,os,sys,json,pickle,numpy as np,pandas as pd,gzip,random
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# In[3]:


def printm(x):
    from IPython.display import display,Markdown
    display(Markdown(x))


# In[33]:


cname='CanonFiction'
DEFAULT_FIELD='Abs-Conc.ALL'
DEFAULT_PERIOD='_median'
EXCLUDE_POS_BESIDES={'n','v','j'}
MIN_N_TEXTS=10
PATH_FIELD_JSON=f'data_counts/{cname}'
FIGDIR='figures'
if not os.path.exists(PATH_FIELD_JSON): os.makedirs(PATH_FIELD_JSON)
badmethods={'WN','ALL','Locke','RH'}
# goodmethods={'PAV-Conc','PAV-Imag','MRC-Conc','MRC-Imag','MT','LSN-Perc','LSN-Imag'}
goodmethods={'PAV-Conc','PAV-Imag','MRC-Conc','MRC-Imag','MT','LSN-Perc','LSN-Imag'}
# goodmethods={'MT'}
goodperiods={}#'C17'}#{'_orig'}


# In[34]:


cutoff=1600
spcr=40
prebreak_cuts=[1500,1000,0,-1000]
prebreaks=[cutoff - (spcr*(i+1)) for i in range(len(prebreak_cuts))]
breaks=[1600,1700,1800,1900,2000]

def edityear4(y,spcr=spcr,cutoff=cutoff,breaks=prebreak_cuts):
    if y>=cutoff: return y
    for i,brk in enumerate(breaks):
        if y>brk:
            brk0=breaks[i-1] if i-1>=0 else cutoff
            return (cutoff-(spcr*(i+1))) + ((y-brk)/(brk0-brk))*spcr


# In[35]:


# countdat
cdf=pd.read_csv(f'data_counts/data.field_counts.{cname}.all_methods_and_periods.csv').set_index('id').dropna()
cdf=cdf[cdf.method.isin(goodmethods)]
if goodperiods: cdf=cdf[cdf.period.isin(goodperiods)]
cdf=cdf.replace([np.inf, -np.inf], np.nan).dropna()
cdf['abs/conc']=[(x/y) if y else np.inf
                 for x,y in zip(cdf['num_abs'],cdf['num_conc'])]# if cdf['num_conc'] else np.nan
# cdf[cdf.period=='_orig']
cdf


# In[36]:


cdf.groupby(['method','period']).median().reset_index()


# In[37]:


cdf.method.value_counts()


# In[38]:


cdf.period.value_counts()


# In[39]:


# cdf[['method','period','perc_conc','perc_abs','perc_neither','abs/conc']]#.groupby(['method','period'])#.apply(zscore,1)


# In[40]:


## Attach metadata
meta=lltk.load(cname).metadata.set_index('id')
# meta


# In[41]:


alldf = cdf.reset_index().groupby('id').mean().join(meta,rsuffix='_meta').fillna('') #,how='inner')
alldf['major_genre']=alldf['major_genre'].apply(lambda x: x if x else 'Unknown')
# alldf['century']=alldf['year'].apply(lambda y: f'C{int(y//100)+1}' if y>1485 else 'Pre-print')
alldf['year0']=alldf.year
alldf['year']=alldf.year.apply(edityear4)

# alldf.query('1500<year0<1600')[['year','year0','canon_genre','major_genre','abs/conc']].sort_values('abs/conc')
alldf


# ## Explore

# In[42]:


# def zscore(col):
#     return [(x - col.std())/col.mean() for x in col]

# for col in tqdm(['perc_abs','perc_conc','abs/conc']):
#     alldf[col+'_z']=zscore(alldf[col])
# alldf


# In[43]:


alldf=alldf[(alldf['canon_genre']!="") | (alldf['corpus_source']!="")]
alldf.loc[alldf['canon_genre'].str.strip()=="", "major_genre"]="Unknown"
alldf


# In[44]:


alldf['dec']=[x//10*10 for x in alldf['year']]
dfplot=alldf.groupby(['major_genre','canon_genre','author']).agg({
    'year':np.median,
    'num_abs':sum,
    'num_conc':sum,
    'num_all':sum,
    'num_neither':sum,
    'perc_conc':np.median,
    'perc_abs':np.median,
    'perc_neither':np.median,
    'abs/conc':np.median
}).reset_index()
dfplot


# ## Plotting

# In[45]:


colors = {'Allegory': '#33a02c',
 'Dialogue': '#1f78b4',
 'Epic': '#b2df8a',
 'Novel': '#a6cee3',
 'Novella': '#fb9a99',
 'Other': '#e31a1c',
 'Pastoral': '#fdbf6f',
 'Picaresque': '#ff7f00',
 'Romance': '#cab2d6',
 'Satire': '#6a3d9a',
 'Tale': '#94945a',
 'Unknown': 'gray',
 'Verse':'#b15928'}


# In[46]:


shapes =  {'Allegory': 'd',
 'Dialogue': '8',
 'Epic': '<',
 'Novel': 'o',
 'Novella': 'v',
 'Other': 'h',
 'Pastoral': 'D',
 'Picaresque': '>',
 'Romance': 's',
 'Satire': 'x',
 'Tale': '+',
 'Unknown': '.',
 'Verse':'*'}


# In[168]:


# interactive
import plotnine as p9
from scipy.stats import zscore
import math
from plotnine import *
factor=2.5

valtype2label={
#     'abs/conc':'<< Concrete words | Abstract words >>',
    'abs/conc':'Frequency of abstract words per 1 concrete word',
    'abs':'% Abstract Words',
    'conc':'% Concrete Words',
    'neither':'% words neither abstract nor concrete',   
}

facet2label = {
    'abs':'Abstract words',
    'conc':'Concrete words',
    'abs/conc':'Abstact / Concrete word ratio'
}


adjust_text_dict = {
    'expand_points': (0, 0),
}

def plot_fiction(
        df=dfplot,
        corpora=[cname],
        field=DEFAULT_FIELD,
        period=DEFAULT_PERIOD,
        color_by='major_genre',
        facet_by='',
        shape_by='major_genre',
        label_by='canon_genre',
        wrap_facet=True,
        valtype='abs/conc',
        color=True,
        span=0.2,
        alpha=1,
        minval=None,
        maxval=None,
        width=9 * factor,
        height=7 * factor,#5.8 * factor,
        dotsize=3,
        standardize=False,
        smooth=True,
        title='',
        save_to=True,
        minyear=0,
        font_size=10,
        jitter=False,
        log_y=False,
        rby=5,
        zrby=0.5,
        highlights=[],
        min_y=None,
        max_y=None,
        spcr=0.5):
    
    # get value
    if valtype=='abs/conc':
        df['value'] = df['abs/conc'] #df['num_abs'] / df['num_conc']
#         df['value'] = 1/df['abs/conc'] ### making it conc/abs!
    elif valtype=='abs':
        df['value'] = df['perc_abs'] #df['num_abs'] / df['num_all'] * 100
    elif valtype=='conc':
        df['value'] = df['perc_conc'] #df['num_conc'] / df['num_all'] * 100
    elif valtype=='neither':
        df['value'] = df['perc_neither'] #df['num_neither'] / df['num_all'] * 100
    else:
        return
    df['value'] = df['value'].apply(lambda y: y if y>min_y else (min_y+((y-min_y)*spcr)))
    df['value'] = df['value'].apply(lambda y: y if y<max_y else max_y)#(max_y+((y-max_y)*(1+spcr))))

    
    # standardize?
    df['zvalue']=zscore([
        np.log10(x) if log_y else x
        for x in df['value']
    ])
    
    ## min max z?
    def padz(z,maxz=2.05,minz=-2.05):
        if z<minz: return minz
        if z>maxz: return maxz
        return z
    df=df.fillna(0)
    df['zvalue'] = df['zvalue'].apply(padz)
    
    df['zbin']=[f'{x//zrby*zrby}' for x in df['zvalue']]
    df['bin']=[f'{str(int(x)).zfill(2)}-{str(int(x+rby)).zfill(2)}' for x in df['value']//rby*rby]# if not standardize else df['value']#int(round(df['value']))
    if standardize: df['value']=df['zvalue']
    df['year']=df.year.apply(lambda y: y if y>minyear else minyear)
    minyear=df['year'].min()
    maxyear=df['year'].max()
    
    if width and height: p9.options.figure_size=(width,height)
    
    # minmax

    # plot
    aes_args={'x':'year','y':'value'}
    if color_by: aes_args['color']=color_by
    if shape_by: aes_args['shape']=shape_by
    aesth=aes(**aes_args)
    
    # start figure
    fig = ggplot(df,aesth)
#     fig+=annotation_stripes(direction='horizontal',fill=['#e3e3e3','#f0f0f0'])
    fig+=theme_classic() 
    fig+=scale_color_manual(colors,show_legend=True,guide='legend')
    fig+=scale_shape_manual(shapes,show_legend=True,guide='legend')
    fig+=scale_x_continuous(
        breaks=prebreaks+breaks,
        labels=[(f'{x*-1} BC' if x<0 else f'{x} AD') if x<=0 else str(x)
                for x in prebreak_cuts+breaks],
    )
    fig+=geom_vline(xintercept=breaks,color='silver')
    fig+=geom_vline(xintercept=prebreaks,color='silver')
    
    

    

    
    
    # set vals
    minval=df['value'].min() if min_y is None else min_y
    maxval=df['value'].max() if max_y is None else max_y
    medianval=df['value'].median()
    stdval=df['value'].std()
    if not standardize:
        fig+=scale_y_continuous(breaks=list(range(0,100,10)))
        if valtype!='abs/conc':
            pass
#             fig+=geom_hline(yintercept = medianval, show_legend=False, color='gray')
#             fig+=geom_hline(yintercept = [10,20,30,40,50], show_legend=False, color='silver')
        else:
            fig+=geom_hline(yintercept = 1, show_legend=False)
        if (minval is not None and maxval is not None):
            fig+=ylim(minval,maxval)
    else:
        fig+=ylim(minval,maxval)# if (minval is None or maxval is None) else ylim(minval,maxval)
        fig+=geom_hline(yintercept = 0.0, show_legend=False)
        
    if dotsize:
        fig+=geom_point(alpha=0.8,size=2,data=df)#b3b3b3')#,show_legend=False)
    
    # labels
    ylabel=valtype2label.get(valtype,valtype)
    def label2facet(x): return facet2label.get(x,x)
    if facet_by: fig+=facet_wrap(facet_by, labeller=label2facet)
    if title: fig+=ggtitle(title)
    if label_by:
        aesd={
            'x':'year',
            'y':'value',
            'label':label_by,
            'guide':False
        }
        dfq=df[(df[label_by]!="") & (df[shape_by]!="" if shape_by else 1)].groupby([x for x in {shape_by,label_by,color_by} if x]).median().reset_index()
        fig+=geom_point(alpha=alpha,size=5,data=dfq)
        if highlights:
            dfl,dfh = dfq[~dfq[label_by].isin(highlights)],dfq[dfq[label_by].isin(highlights)]
        else:
            dfl,dfh = dfq,None
        if not jitter:
            fig+=geom_text(aes(**aesd),inherit_aes=False,data=dfl)#,adjust_text=adjust_text_dict)
        else:
            fig+=geom_text(aes(**aesd),inherit_aes=False,adjust_text=adjust_text_dict,data=dfl)    
        if dfh is not None:
            fig+=geom_text(aes(**aesd),fontweight='bold',color='black',data=dfh)
    
    fig+=ylab(ylabel)
    fig+=xlab('Year')
    fig+=guides(fill = False)#, color = True, linetype = False, shape = False)
    
    if log_y:
        if valtype!='abs/conc':
            fig+=scale_y_log10()
        else:
            fbrks=[1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100]
            fig+=scale_y_continuous(trans='log2',breaks=[_x/10 for _x in fbrks])
    else:
        if valtype !='abs/conc':
#             fig+=scale_y_continuous(breaks=list(range(0,105,5)))
            fig+=scale_y_continuous(breaks=[0,10,20,30,40,50,60,70,80,90,100])
            
    if smooth:
        fig+=geom_smooth(
            aes(x='year',y='value'),
            inherit_aes=False,
            span=span,
            se=True,
            method='loess',
            alpha=0.15,
            color='gray',
            data=df
        )
    
    if save_to:
        if save_to is True:
            save_to=os.path.join(FIGDIR, f'fig.absrealism.{corpora[0]}.{field}.{period}.{valtype.replace("/","_")}{".clean" if jitter else ""}.v26.png')
            
        save_to_dir=os.path.dirname(save_to)
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        fig.save(save_to)
    
    return fig


# ## Plots

# In[169]:


JITTER=0


# In[170]:


## Combos

def do_plot_fiction(**x):
    args={
        **dict(
            valtype='abs/conc',
            title='Prevalence of abstract vs. concrete words across history of fictional canon',
            jitter=0,
            standardize=False,
            log_y=False,
            color_by='major_genre',
#             color_by=None,
#             color_by='annotated',
            highlights={'Austen','Cusk'},
        ),
        **x
    }
    return plot_fiction(**args)


# In[171]:


# import seaborn as sns
# pal = sns.color_palette('Paired', 12)
# hexes=list(pal.as_hex())
# pal.as_hex()


# In[172]:


# for i,c in enumerate(colors): colors[c]=hexes[i]
# colors


# In[173]:


# do_plot_fiction(valtype='abs/conc',title='Ratio of abstract to concrete words in fiction',
#                 max_y=10,min_y=0.15)
# do_plot_fiction(valtype='abs/conc',title='Ratio of abstract to concrete words in fiction',
#                 max_y=11,min_y=0.3,jitter=1)


# In[181]:


do_plot_fiction(
    valtype='conc',
    title='Density of concrete words in fiction',
    min_y=0,
    max_y=55,
    jitter=1
)


# In[182]:


do_plot_fiction(
    valtype='abs',
    title='Density of abstract words in fiction',
    min_y=11,
    max_y=69,
    jitter=1
)


# In[166]:


# do_plot_fiction(valtype='abs',title='Proportion of words concrete',
#                 max_y=60,min_y=0,log_y=False,jitter=1)


# In[183]:


# do_plot_fiction(valtype='neither',title='Ratio of abstract to concrete words in fiction',
#                 max_y=60,min_y=0,log_y=False)


# In[ ]:





# In[184]:


# do_plot_fiction(valtype='conc',title='Density of concrete language in fiction',log_y=False,span=.2)
# do_plot_fiction(valtype='conc',title='Density of concrete language in fiction',log_y=False,jitter=1)


# In[185]:


# do_plot_fiction(valtype='abs',title='Density of abstract language in fiction',log_y=False,span=.2)
# do_plot_fiction(valtype='abs',title='Density of abstract language in fiction',log_y=False,jitter=1)


# In[29]:


# do_plot_fiction(valtype='neither',title='Density of neutral language in fiction',log_y=False)


# In[30]:


# ## Combos

# # both
# printm('### Abstract/Concrete ratio (clean)')
# display(plot_fiction(
#     df=dfq,
#     valtype='abs/conc',
# #     title='Ratio of abstract to concrete words across fictional canon',
#     title='Prevalence of abstract vs. concrete words across history of fictional canon',
#     jitter=1,
#     standardize=False,
#     log_y=True,
#     color_by=None
    
# ))
# printm('----')


# In[31]:


# # Concrete
# printm('## Plots')
# printm('### Concrete words')
# display(plot_fiction(
#     df=dfq,
#     valtype='conc',
#     title='Prevalence of concrete words across fictional canon',
#     jitter=JITTER,
# ))
# printm('----')

# # Abstract
# printm('### Abstract words')
# display(plot_fiction(
#     df=dfq,
#     valtype='abs',
#     title='Prevalence of abstract words across fictional canon',
#     jitter=JITTER
# ))
# printm('----')

# # neither
# printm('### Neutral words')
# display(plot_fiction(
#     df=dfq,
#     valtype='neither',
#     title='Prevalence of neutral words across fictional canon',
#     jitter=JITTER
# ))
# printm('----')


# In[ ]:





# In[ ]:





# In[ ]:




