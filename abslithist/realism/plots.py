import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..'))
from abslithist import *
from abslithist.words import *

JITTER=0
VERSION='v36'

factor=2.5
cutoff=1600
spcr=40
prebreak_cuts=[1500,1000,0,-1000]
prebreaks=[cutoff - (spcr*(i+1)) for i in range(len(prebreak_cuts))]
breaks=[1600,1700,1800,1900,2000]


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


valtype2label={
    # 'abs/conc':'<< Concrete words | Abstract words >>',
    'abs-conc':'<< More concrete words | More abstract words >>   ',
    # 'abs-conc':'# Abstract words - # Concrete words (averaged across all 100-word passages)',
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







# function to compact years
def edityear(y,spcr=spcr,cutoff=cutoff,breaks=prebreak_cuts):
    if y>=cutoff: return y
    for i,brk in enumerate(breaks):
        if y>brk:
            brk0=breaks[i-1] if i-1>=0 else cutoff
            return (cutoff-(spcr*(i+1))) + ((y-brk)/(brk0-brk))*spcr


# loading data
def load_data_for_plotting(cname='CanonFiction',sources=SOURCES_FOR_PLOTTING,periods={}):
    # countdat
    #cdf=pd.read_csv(f'data/counts/data.absconc.{cname}.csv')#.set_index('id').dropna()
    #cdf.to_feather(f'data/counts/data.absconc.{cname}.ft')
    cdf=pd.read_feather(f'data/counts/data.absconc.{cname}.v6.csv.ft')
    # cdf=pd.read_csv(f'{COUNT_DIR}/data.absconc.{cname}.psgs.v7.csv.gz')
    if sources: cdf=cdf[cdf.source.isin(sources)]
    if periods: cdf=cdf[cdf.period.isin(periods)]
    cdf['abs/conc']=cdf['num_abs']/cdf['num_conc']
    for key in ['abs','conc','neither']:
        cdf['perc_'+key]=cdf['num_'+key]/cdf['num_total']

    # attach meta
    import lltk
    meta=lltk.load(cname).metadata
    alldf = cdf.merge(meta,on='id',how='inner')

    # clean
    alldf['major_genre']=alldf['major_genre'].apply(lambda x: x if x else 'Unknown')
    alldf['year_orig']=alldf['year']
    alldf['year']=alldf['year_orig'].apply(edityear)
    alldf['dec']=[x//10*10 for x in alldf['year']]

    # filter
    alldf=alldf[(alldf['canon_genre']!="") | (alldf['corpus_source']!="")]
    alldf.loc[alldf['canon_genre'].str.strip()=="", "major_genre"]="Unknown"

    dfplot=alldf.groupby(['major_genre','canon_genre','author']).mean().reset_index().sort_values('abs/conc')
    # dfplot=alldf.groupby(['major_genre','canon_genre','author']).median().reset_index().sort_values('abs/conc')

    return dfplot











adjust_text_dict = {
    'expand_points': (0, 0),
}

def plot_fiction(
        df,
        corpora=['CanonFiction'],
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
    elif valtype=='abs-conc':
        df['value'] = df['num_abs']-df['num_conc'] #df['num_abs'] / df['num_conc']
    elif valtype=='abs+conc':
        df['value'] = df['num_abs']+df['num_conc'] #df['num_abs'] / df['num_conc']
    elif valtype=='abs':
        df['value'] = df['perc_abs']*100 #df['num_abs'] / df['num_all'] * 100
    elif valtype=='conc':
        df['value'] = df['perc_conc']*100 #df['num_conc'] / df['num_all'] * 100
    elif valtype=='neither':
        df['value'] = df['perc_neither']*100 #df['num_neither'] / df['num_all'] * 100
    else:
        return
    df=df[df['value']!=None]
    df['value'] = df['value'].apply(lambda y: y if y>min_y else (min_y+((y-min_y)*spcr)))
    df['value'] = df['value'].apply(lambda y: y if y<max_y else max_y)#(max_y+((y-max_y)*(1+spcr))))

    
    # standardize?
    if standardize:
        if valtype=='abs/conc':
            df['zvalue']=zscore([
                np.log10(x) if log_y else x
                for x in df['value']
            ])
        else:
            df['zvalue']=zscore(df['value'])
    
        ## min max z?
        def padz(z,maxz=max_y,minz=min_y):
            if z<minz: return minz
            if z>maxz: return maxz
            return z
        df=df.fillna(0)
        # df['zvalue'] = df['zvalue'].apply(padz)
        df['value']=df['zvalue']

    
    df['year']=df.year.apply(lambda y: y if y>minyear else minyear)
    minyear=df['year'].min()
    maxyear=df['year'].max()
    
    if width and height: p9.options.figure_size=(width,height)
    
    # minmax

    # START FIGURE

    aes_args={'x':'year','y':'value'}
    if color_by: aes_args['color']=color_by
    if shape_by: aes_args['shape']=shape_by
    aesth=p9.aes(**aes_args)
    
    # start figure
    fig = p9.ggplot(df,aesth)
#     fig+=p9.annotation_stripes(direction='horizontal',fill=['#e3e3e3','#f0f0f0'])
    fig+=p9.theme_classic() 
#     fig+=p9.theme(
# #         text=element_text(fontproperties=body_text),
#         axis_title_x=p9.element_text(family='monospace'),
#         axis_title_y=p9.element_text(family='monospace'),
#         axis_text_x=p9.element_text(family='monospace'),
#         axis_text_y=p9.element_text(family='monospace')
#     )
    fig+=p9.scale_color_manual(colors,show_legend=True,guide='legend')
    fig+=p9.scale_shape_manual(shapes,show_legend=True,guide='legend')
    fig+=p9.scale_x_continuous(
        breaks=prebreaks+breaks,
        labels=[(f'{x*-1} BC' if x<0 else f'{x} AD') if x<=0 else str(x)
                for x in prebreak_cuts+breaks],
    )
    fig+=p9.geom_vline(xintercept=breaks,color='silver')
    fig+=p9.geom_vline(xintercept=prebreaks,color='silver')
    
    

    

    
    
    # set vals
    minval=df['value'].min() if min_y is None else min_y
    maxval=df['value'].max() if max_y is None else max_y
    medianval=df['value'].median()
    stdval=df['value'].std()
    if not standardize:
        fig+=p9.scale_y_continuous(breaks=list(range(0,100,10)),limits=(min_y,max_y))
        if valtype=='abs/conc':
            fig+=p9.geom_hline(yintercept = 1, show_legend=False,color='gray')
        elif valtype=='abs-conc':
            fig+=p9.geom_hline(yintercept=0,show_legend=False,color='gray')
        else:
            fig+=p9.geom_hline(yintercept = medianval, show_legend=False, color='gray')
            pass
        if (minval is not None and maxval is not None):
            fig+=p9.ylim(minval,maxval)
    else:
        fig+=p9.ylim(minval,maxval)
        fig+=p9.geom_hline(yintercept = 0.0, show_legend=False)
        
    if dotsize:
        fig+=p9.geom_point(alpha=0.5,size=2,data=df)#b3b3b3')#,show_legend=False)
    
    # labels
    ylabel=valtype2label.get(valtype,valtype)
    def label2facet(x): return facet2label.get(x,x)
    if facet_by: fig+=p9.facet_wrap(facet_by, labeller=label2facet)
    if title: fig+=p9.ggtitle(title)
    if label_by:
        aesd={
            'x':'year',
            'y':'value',
            'label':label_by,
            'guide':False
        }
        dfq=df[(df[label_by]!="") & (df[shape_by]!="" if shape_by else 1)].groupby([x for x in {shape_by,label_by,color_by} if x]).median().reset_index()
        fig+=p9.geom_point(alpha=alpha,size=5,data=dfq)
        if highlights:
            dfl,dfh = dfq[~dfq[label_by].isin(highlights)],dfq[dfq[label_by].isin(highlights)]
        else:
            dfl,dfh = dfq,None
        fig+=p9.geom_text(
            p9.aes(**aesd),
            inherit_aes=False,
            data=dfl,
            adjust_text=adjust_text_dict if jitter else None,
        )
        if dfh is not None:
            fig+=p9.geom_text(p9.aes(**aesd),fontweight='bold',color='black',data=dfh)
    
    fig+=p9.ylab(ylabel)
    fig+=p9.xlab('Year')
    fig+=p9.guides(fill = False)#, color = True, linetype = False, shape = False)
    
    if log_y:
        if valtype!='abs/conc':
            fig+=p9.scale_y_log10()
        else:
            fbrks=[1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100]
            fig+=p9.scale_y_continuous(trans='log2',breaks=[_x/10 for _x in fbrks])
    else:
        if valtype in {'abs','conc','neither'}:
#             fig+=scale_y_continuous(breaks=list(range(0,105,5)))
            fig+=p9.scale_y_continuous(breaks=[0,10,20,30,40,50,60,70,80,90,100])
        elif valtype=='abs-conc':
            fig+=p9.scale_y_continuous(breaks=[-50,-40,-30,-20,-10,0,10,20,30,40,50],limits=[min_y-2,max_y+2])
            # fig+=p9.scale_y_continuous(breaks=[-400,-300,-200,-100,0,100,200,300,400])
        else:
            fig+=p9.scale_y_continuous()


    if smooth:
        fig+=p9.geom_smooth(
            p9.aes(x='year',y='value'),
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
            save_to=os.path.join('figures', f'fig.absrealism.{corpora[0]}.{valtype.replace("/","_")}{".clean" if jitter else ""}.{VERSION}.png')
            
        save_to_dir=os.path.dirname(save_to)
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        fig.save(save_to)
    
    return fig


# ## Plots







## Combos

def do_plot_fiction(df=None,**x):
    if df is None: df=load_data_for_plotting()
    args={
        **dict(
            valtype='abs/conc',
            title='# Abstract words - # Concrete words, averaged across all passages containing 100 recognized words',
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
    return plot_fiction(df,**args)

if __name__=='__main__':
    print('>> Loading data')
    df=load_data_for_plotting() #periods={'C17'})#
    print('done')

    # do_plot_fiction(df,valtype='abs/conc',standardize=False,log_y=True,max_y=3,min_y=-1)     
    do_plot_fiction(df,valtype='abs-conc',standardize=False)
    # do_plot_fiction(df,valtype='abs')
    # do_plot_fiction(df,valtype='conc')
    # do_plot_fiction(df,valtype='abs/conc',standardize=False,log_y=True,max_y=3,min_y=-1)     
    # do_plot_fiction(df,valtype='neither')



    # do_plot_fiction(
    #     valtype='conc',
    #     title='Density of concrete words in fiction',
    #     min_y=0,
    #     max_y=55,
    #     jitter=1
    # )





    # do_plot_fiction(
    #     valtype='abs',
    #     title='Density of abstract words in fiction',
    #     min_y=11,
    #     max_y=69,
    #     jitter=1
    # )


