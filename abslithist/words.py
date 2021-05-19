import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..'))
from abslithist import *


STOPWORDS=None
SPELLING_D=None

REPLACEMENTS={
    '&hyphen;':'-',
    '&sblank;':'--',
    '&mdash;':' -- ',
    '&ndash;':' - ',
    '&longs;':'s',
    '|':'',
    '&ldquo;':u'“',
    '&rdquo;':u'”',
    '&lsquo;':u'‘’',
    '&rsquo;':u'’',
    '&indent;':'     ',
    '&amp;':'&',
}
REPLACEMENTS_b = {
    0x2013: ' -- ',
    0x2014: ' -- ',
    0x201c: '"',
    0x201d: '"',
    0x2018: "'",
    0x2019: "'",
    0x2026: ' ... ',
    0xa0: ' '
}


def get_stopwords(lower=True):
    global STOPWORDS
    if not STOPWORDS:
        paths = [
            path for path in 
            [PATH_STOPWORDS,PATH_NAMES]
            if os.path.exists(path)
        ]
        stopwords=set()
        for path in paths:
            with open(path) as f:
                words=f.read().strip().split('\n')
                stopwords|=set((w.strip().lower() if lower else w.strip()) for w in words if w.strip())
            # print(path,len(stopwords))
        STOPWORDS=stopwords
    return STOPWORDS

def get_spelling_modernizer(path=PATH_SPELLING_D):
    global SPELLING_D
    if not SPELLING_D:
        with open(path) as f:
            SPELLING_D=dict([
                ln.strip().split('\t',1)
                for ln in f
                if ln.strip() and '\t' in ln and not ln.startswith('#')
            ])
    return SPELLING_D


###
# loading models
###



SENTENCE_TOKENIZER=None

def get_sentence_tokenizer():
    global SENTENCE_TOKENIZER
    if SENTENCE_TOKENIZER is None:
        import stanza
        SENTENCE_TOKENIZER = stanza.Pipeline(lang='en', processors='tokenize')
    return SENTENCE_TOKENIZER

def tokenize_sentences_nlp(txt):
    nlp = get_sentence_tokenizer()
    return [
        #[token.text for token in sentence.tokens]
        sentence.text
        for sentence in nlp(txt).sentences
    ]

def tokenize_sentences(txt):
    import nltk
    return nltk.sent_tokenize(txt)

def gleanPunc2(aToken):
    aPunct0 = ''
    aPunct1 = ''
    while(len(aToken) > 0 and not aToken[0].isalnum()):
        aPunct0 = aPunct0+aToken[:1]
        aToken = aToken[1:]
    while(len(aToken) > 0 and not aToken[-1].isalnum()):
        aPunct1 = aToken[-1]+aPunct1
        aToken = aToken[:-1]

    return (aPunct0, aToken, aPunct1)

def modernize_spelling_in_txt(txt,spelling_d):
    lines=[]
    for ln in txt.split('\n'):
        ln2=[]
        for tok in ln.split(' '):
            p1,tok,p2=gleanPunc2(tok)
            tok=spelling_d.get(tok,tok)
            ln2+=[p1+tok+p2]
        ln2=' '.join(ln2)
        lines+=[ln2]
    return '\n'.join(lines)


def tokenize(txt,lower=True,modernize=MODERNIZE_SPELLING):
    # print(lower,modernize,'hi')
    # clean
    for k,v in REPLACEMENTS_b.items(): txt = txt.replace(chr(k), v)
    for k,v in REPLACEMENTS.items(): txt = txt.replace(k, v)
    # modernize?
    txt=txt.lower() if lower else txt
    if modernize:
        spellingd=get_spelling_modernizer()
        txt=modernize_spelling_in_txt(txt,spellingd)
    # tokenize
    words = nltk.word_tokenize(txt)
    return words

def tokenize_fast(line,lower=True,incl_punct=False):
    import re
    from string import punctuation
    line=line.lower() if lower else line
    tokens = re.findall(
        r".*?[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
        # r'\w+',
        line
    )
    if not incl_punct: tokens = [w.strip(punctuation) for w in tokens]
    tokens = [w for w in tokens if w]
    return tokens





# Parse raw text into paras/sentences

def to_sents(txt,words_recog=set(),num_word_min=45,vald={},valname='val',sep_para='\n\n',stopwords=set(),modernize_spelling=True,correct_ocr=True):
    ntok,nword,npara,nsent=0,-1,0,0
    if not stopwords: stopwords=get_stopwords()
    if modernize_spelling: modd=get_spelling_modernizer()
    if correct_ocr: ocrd=get_ocr_corrections()
    if not words_recog: words_recog=get_wordlist()
    txt=txt.strip()
    o=[]
    
    def _cleantok(x):
        return {
            ' ':'_',
            '\n':'|'
        }.get(x,x)
    
    paras=txt.split(sep_para) if sep_para else [txt]
    for pi,para in enumerate(paras):
        para=para.strip()
        for si,sent in enumerate(tokenize_sentences(para)):
            sent=sent.strip()
            swords=tokenize_agnostic(sent)
            # swords=[_cleantok(x) for x in swords]
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
                    'tokl_mod':modd.get(wl,wl) if modernize_spelling else wl,
                    'tokl_ocr':ocrd.get(wl,wl) if correct_ocr else wl,
                    'is_punct':ispunc,
                }
                if words_recog:
                    dx['is_recog']=int(any(dx[k] in words_recog for k in ['tok','tokl','tokl_mod','tokl_ocr']))
                    if len(wl)<2 and wl!='i': dx['is_recog']=0
                if vald: dx[valname]=vald.get(wl,np.nan)
                o+=[dx]
                ntok+=1
            nsent+=1
        npara+=1
    odf=pd.DataFrame(o)
    odf=odf.set_index('i_tok') if 'i_tok' in odf.columns else odf
    if 'tokl' in odf.columns:
        odf['is_stopword']=odf.tokl.apply(lambda x: int(x in stopwords))
    return odf
