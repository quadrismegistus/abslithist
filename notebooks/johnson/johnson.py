import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, "/Users/ryan/github/lltk")
from lltk import *
from abslithist import *
from abslithist.words import *
import spacy
from functools import lru_cache as cache
from diskcache import Cache

cache_obj = Cache(os.path.join(os.path.dirname(__file__), "cache.dc"))

nlp = spacy.load("en_core_web_sm")


def get_head_foot(sent, tok):
    head = tok.head
    foot = None
    for w in sent:
        if w.head == tok or w.head == tok.head and w.pos_ not in {"PUNCT"}:
            foot = w
    return head, foot


def find_phrase_subtree(sent, tok, stop=None):
    try:
        o = [x for x in tok.subtree if (stop is None or x.i < stop)]
        o = sent[min(o).i : max(o).i + 1]
        return tuple(o)
    except (ValueError, IndexError):
        return ()


def find_phrase_span(sent, tok, stop=None):
    try:
        o = [
            x
            for x in sent
            if (x == tok or x.head == tok)
            and (stop == None or x.i < stop)
            and x.i >= tok.i
        ]
        o = sent[min(o).i : max(o).i + 1]
        return tuple(o)
    except (ValueError, IndexError):
        return ()


def phrases_are_parallel(
    phr1,
    phr2,
    tok,
    pos_aggregation={
        "PROPN": "NOUN",
        "CCONJ": "CONJ",
        "SCONJ": "CONJ",
    },
    require_linearity=True,
):
    try:
        il = [x.i for x in phr1] + [tok.i] + [x.i for x in phr2]
        il2 = list(range(phr1[0].i, phr2[-1].i + 1))
        if not require_linearity or il == il2:
            phr1 = [x for x in phr1 if x.pos_ not in {"PUNCT"}]
            phr2 = [x for x in phr2 if x.pos_ not in {"PUNCT"}]

            phr1x = [x for x in phr1]
            phr2x = [x for x in phr2]
            while phr1x and phr1x[0].pos_ in {"PRON", "DET"}:
                phr1x = phr1x[1:]
            while phr2x and phr2x[0].pos_ in {"PRON", "DET"}:
                phr2x = phr2x[1:]
            l1 = [pos_aggregation.get(x.pos_, x.pos_) for x in phr1x]
            l2 = [pos_aggregation.get(x.pos_, x.pos_) for x in phr2x]
            if l1 == l2:
                return (tuple(phr1), tuple(phr2))
    except (ValueError, IndexError):
        pass
    return None, None


def find_parallel_phrases_in_sent(
    sent, deps={"cc", "prep"}, minlen=None, as_str=False, require_linearity=True
):
    o = []
    sent = nlp(sent) if type(sent) == str else sent
    for w in sent:
        if w.dep_ in deps:
            head, foot = get_head_foot(sent, w)
            if head and foot:
                headphr, footphr = find_phrase_subtree(
                    sent, head, stop=w.i
                ), find_phrase_subtree(sent, foot)
                headphr, footphr = phrases_are_parallel(headphr, footphr, w)
                if headphr and footphr:
                    o.append([headphr, w, footphr])
                else:
                    headphr, footphr = find_phrase_span(
                        sent, head, stop=w.i
                    ), find_phrase_span(sent, foot)
                    headphr, footphr = phrases_are_parallel(
                        headphr, footphr, w, require_linearity=require_linearity
                    )
                    if headphr and footphr:
                        o.append([headphr, w, footphr])
                    # if len(headphr)>1:
                    # print(o[-1])
    if minlen:
        o = [phr for phr in o if len(phr[0]) >= minlen]
    if as_str:

        def to_str(x):
            return tuple([(y.text, y.pos_) for y in x])

        o = [(to_str(x), (y.text, y.pos_), to_str(z)) for x, y, z in o]
    return o


def get_db(flag="c", suffix="2"):
    from sqlitedict import SqliteDict

    path = os.path.join(os.path.dirname(__file__), f"cache{suffix}.db")
    return SqliteDict(path, flag=flag, autocommit=True)


def find_parallel_phrases(
    txt, deps=("cc", "prep"), minlen=None, lim_sents=1000, require_linearity=True
):
    o = []
    txt = " ".join(txt.strip().split())
    for i, sent in enumerate(tokenize_sentences(txt)):
        if i >= lim_sents:
            break
        sentphr = find_parallel_phrases_in_sent(
            sent,
            deps=deps,
            minlen=minlen,
            as_str=True,
            require_linearity=require_linearity,
        )
        if sentphr:
            o.append((i, sent, sentphr))
    return o


def get_parallel_phrase_quotient(txt, minlen=None):
    phrases = find_parallel_phrases(txt, minlen=minlen)
    try:
        return len(phrases) / len(txt.split()) * 100
    except Exception:
        return np.nan


def process_folder(
    path_texts, force=False, minlen=None, require_linearity=True, suffix="2"
):
    db = get_db(flag="c", suffix=suffix)
    files = list(sorted(os.listdir(path_texts)))
    random.shuffle(files)
    for fn in tqdm(files):
        fnid = os.path.splitext(fn)[0]
        if not force and fnid in db:
            continue
        fnfn = os.path.join(path_texts, fn)
        with open(fnfn) as f:
            txt = f.read()
        vl = find_parallel_phrases(
            txt, minlen=minlen, require_linearity=require_linearity
        )
        db[fnid] = vl


def process_corpus(
    corpus, ids=None, force=False, minlen=None, require_linearity=True, suffix="2"
):
    db = get_db(flag="c", suffix=suffix)
    if not ids:
        ids = list(corpus.textd.keys())
    random.shuffle(ids)
    pbar = tqdm(ids)
    for fnid in pbar:
        if not force and fnid in db:
            continue
        t = corpus.textd[fnid]
        vl = find_parallel_phrases(
            t.txt, minlen=minlen, require_linearity=require_linearity
        )
        pbar.set_description(f"{fnid}: {len(vl)} sents with parallels")
        db[fnid] = vl


def remove_opening_determiners(posx):
    while posx and posx[0] in {"DET", "PRON"}:
        posx = posx[1:]
    return posx


def unpack_parallel(xyz):
    x, y, z = xyz
    words = (tuple(a[0] for a in x), y[0], tuple(b[0] for b in z))
    pos = (tuple(a[1] for a in x), y[1], tuple(b[1] for b in z))
    words_flat = tuple(list(words[0]) + [words[1]] + list(words[2]))
    is_valid = not all(w == w.upper() for w in words_flat)

    plen = min(
        [
            len(remove_opening_determiners(pos[0])),
            len(remove_opening_determiners(pos[-1])),
        ]
    )
    midword = words[1]  # .lower()
    # if midword=='&': midword='and'
    return {
        "word_beg": " ".join(words[0]),
        "word_mid": midword,
        "word_end": " ".join(words[2]),
        "pos_beg": " ".join(pos[0]),
        "pos_mid": pos[1],
        "pos_end": " ".join(pos[2]),
        "plen": plen,
        "is_valid": is_valid,
    }


def get_db_data(suffix=""):
    db = get_db(flag="r", suffix=suffix)
    o = []
    for key in tqdm(db.keys(), total=len(db)):
        for sent_num, sent, sent_parallels in db[key]:
            for paral in sent_parallels:
                pdat = unpack_parallel(paral)
                if pdat["is_valid"]:
                    o.append({"id": key, "sent_num": sent_num, **pdat, "sent": sent})
    return pd.DataFrame(o).set_index("id")


path_texts = "/Users/ryan/lltk_data/corpora/wimsatt/texts"
