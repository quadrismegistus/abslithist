import os,sys; sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..'))
from abslithist import *


STOPWORDS=None
SPELLING_D=None

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



def tokenize(txt,lower=True,modernize=False):
	replacements = {
		0x2013: ' -- ',
		0x2014: ' -- ',
		0x201c: '"',
		0x201d: '"',
		0x2018: "'",
		0x2019: "'",
		0x2026: ' ... ',
		0xa0: ' '
	}
	for r in replacements:
		txt = txt.replace(chr(r), replacements[r])
	
	words = nltk.word_tokenize(txt)
	if modernize:
		spellingd=get_spelling_modernizer()
		words = [spellingd.get(w,w) for w in words]
	return words

def tokenize_fast(line,lower=True):
	import re
	from string import punctuation
	line=line.lower() if lower else line
	tokens = re.findall(
		r".*?[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
		# r'\w+',
		line
	)
	tokens = [w.strip(punctuation) for w in tokens]
	tokens = [w for w in tokens if w]
	return tokens




