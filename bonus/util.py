import json
import spacy
import gensim
import numpy as np
from tqdm import tqdm

def buildDictionary(document, w2i_path, i2w_path, src_path):
	nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'parser', 'textcat'])
	embedd_model = gensim.models.KeyedVectors.load_word2vec_format(src_path, binary = True, unicode_errors = 'ignore')
	special = ['<pad>', '<s>', '</s>', '<unk>']
	pipe = nlp.pipe(document)
	words = []
	for doc in tqdm(pipe):
		words += [token.text.lower() for token in doc if token.text.lower() in embedd_model.vocab]
	#print(len(words))
	words = set(words)
	words = special+list(words)
	print(len(words))
	word2index = {}
	index2word = []
	#print(words)
	for i, w in enumerate(words):
		word2index[w] = i
		index2word.append(w)

	f = open(w2i_path, "w")
	json.dump(word2index, f)
	f.close()
	f = open(i2w_path, "w")
	json.dump(index2word, f)
	f.close()
	return  words

def buildWordEmbedding(words, src_path, embedding_path):
	embedd_model = gensim.models.KeyedVectors.load_word2vec_format(src_path, binary = True, unicode_errors = 'ignore')
	embedding =[]
	for w in tqdm(words):
		try:
			vec = embedd_model[w]
		except KeyError:
			vec = np.random.normal(size=300)
		embedding.append(vec)
	np.save(embedding_path, np.array(embedding))
	return np.array(embedding)

def padding(seq, to_len, padding):
	padd = []
	for s in seq:
		padd.append(s[:to_len] + [padding]*max(0, to_len - len(s)))
	return padd
