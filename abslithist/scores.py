from abslithist import *
VERSION='v6'


# func

def to_scores(sentdf_or_txt,norms=None,source='Median',period='median',tokname='tokl',valname='val',nmin=50,stopwords=set(),sep_para='\n\n',w2score=None,w2score_perc=None):
    # parse sents
    sentdf = to_sents(sentdf_or_txt,sep_para=sep_para) if type(sentdf_or_txt)==str else sentdf_or_txt

    # get norms
    if w2score is None or w2score_perc is None:
        w2score,w2score_perc=get_norm_dict(norms=norms,source=source,period=period,stopwords=stopwords,remove_stopwords=True)

    # set value
    sentdf[valname]=sentdf[tokname].apply(lambda x: w2score.get(x))
    sentdf[valname+'_perc']=sentdf[tokname].apply(lambda x: w2score_perc.get(x))
    sentdf['is_recog']=sentdf[valname].apply(lambda x: int(not np.isnan(x)))
    return sentdf#.set_index('i_tok')


def score_freqs(freqd_or_path_freqs,w2score,w2score_perc={}):
    # print('computing score freqs')
    if type(freqd_or_path_freqs)==str: 
        with open(freqd_or_path_freqs) as f: freqd=json.load(f)
    else:
        freqd=freqd_or_path_freqs
    freqd2=dict((w,c) for w,c in freqd.items() if w in w2score)
    summ=sum(freqd2.values())
    tf = dict((w,c/summ) for w,c in freqd2.items())
    odx={}
    odx['val']=sum((tf.get(w,0) * w2score.get(w,0)) for w in tf)
    odx['val_perc']=sum((tf.get(w,0) * w2score_perc.get(w,0)) for w in tf)
    # print('done')
    return odx



def do_score_text(inp):
    path_txt,path_freqs,odir,period,ometa,num_proc,nmin,freqs_only=inp
    ofn_scores = os.path.join(odir,'scores.pkl')
    ofn_psgs = os.path.join(odir,'passages.pkl')
    ofn_freqscore = os.path.join(odir,'freqscore.pkl')
    if not os.path.exists(odir): os.makedirs(odir)
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


def score_corpus(C,force=False,num_proc=DEFAULT_NUM_PROC,nmin=50,freqs_only=False):
    # objs
    objs = [
        (
            t.path_txt,
            t.path_freqs,
            os.path.join(PATH_SCORES_BYTEXT,t.corpus.name,t.id),
            to_field_period(t.year),
            {'id':t.id, 'corpus':t.corpus.name},
            1,#num_proc,
            nmin,
            freqs_only
        ) for t in C.texts()
    ]
    # Do all
    res=pd.DataFrame(pmap(
        do_score_text,
        objs,
        num_proc=num_proc,
        desc='Scoring passages',
    ))
    save_df(res, os.path.join(PATH_SCORES_BYTEXT,C.name+'.pkl'))
    return res


def get_current_text_scores(corpora=[DEFAULT_CORPUS], incl_meta=['author','title','year','canon_genre']):
    ifns=[os.path.join(PATH_SCORES_BYTEXT,cname+'.pkl') for cname in corpora]
    df=pd.concat(pd.read_pickle(ifn) for ifn in ifns)
    df['val']=df[[x for x in df.columns if x.endswith('_val')]].mean(axis=1)
    df['val_perc']=df[[x for x in df.columns if x.endswith('_val_perc')]].mean(axis=1)
    df=df.set_index(['corpus','id'])
    
    # join with meta?
    dfmeta=corpora_meta(corpora,incl_meta=incl_meta)
    odf=df.join(dfmeta)
    odf=odf.dropna().sort_values('val_perc')
    return odf