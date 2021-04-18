import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..'))
from abslithist import *

VERSION='v6'
DEFAULT_CORPUS='CanonFiction'
SLINGSHOT_FN=os.path.abspath('.sling.score_freqs.py')
CORPUS=sys.argv[1] if len(sys.argv)>1 and not sys.argv[1].startswith('-') else DEFAULT_CORPUS

slingshot_py="""
import sys
sys.path.append('../..')
from abslithist.words import *
import ujson as json

dfnorms=get_allnorms()

def score_freqs(inpd):
    path_freqs=inpd['path_freqs']
    try:
        with open(path_freqs) as f: freqs=json.load(f)
    except Exception as e:
        print('!',e)
        return []
    freql=[w for w,c in freqs.items() for i in range(c)]
    freqdf=pd.DataFrame(index=freql)
    
    # by col
#     odf=pd.DataFrame()
#     for col in dfnorms.columns:
#         scol=dfnorms[col]
#         scoldf=freqdf.join(scol.dropna().to_frame(),how='inner')
#         # make avg
#         avgdf=pd.DataFrame(dict((x,scoldf.agg(x)) for x in ['mean','median','sum','std','size']))
#         odf=odf.append(avgdf)
#     return odf.rename_axis('norm').reset_index().to_dict('records')
    bothdf=freqdf.join(dfnorms)
    return dict(bothdf.mean())
"""
def get_savedir(C):
    return os.path.join(PATH_DATA,'scores',VERSION,C.name)

def score_corpus(C,savedir=None,**attrs):
    # get objects
    objects = [
        {'id':t.id, 'path_freqs':t.path_freqs}
        for t in C.texts()
    ][:1000]
    

    # Slingshot!s
    savedir=get_savedir(C) if not savedir else savedir
    with open(SLINGSHOT_FN,'w') as of: of.write(slingshot_py)
    cmd = sl.shoot(
        func='score_freqs',
        path_src=SLINGSHOT_FN,
        objects=objects,
        savedir=savedir,
        parallel=DEFAULT_NUM_PROC,
        **attrs
    #     overwrite=True
    )
    os.system(cmd)
    return gather_scores_corpus(C,savedir)


def gather_scores_corpus(C,savedir=None):
    scorefn=os.path.join(PATH_DATA,'scores',VERSION,C.name+'.pkl')
    if not os.path.exists(scorefn):
        old=[]
        savedir=get_savedir(C) if not savedir else savedir
        cachedir=os.path.join(savedir,'cache')
        for idx,dx in sl.stream_results(cachedir):
            # print(metadx,dx)
            if type(dx)!=dict: continue
            for k,v in dx.items():
                odx={'id':idx}
                odx['contrast'],odx['source'],odx['period']=k.split('.')
                old.append(odx)
        df=pd.DataFrame(old)
        save_df(df,scorefn)
        return df
    return read_df(scorefn)