import os
import logging
import json
import pickle
import buildS2SDatasetForTest
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils import data
from model import Seq2Seq, Encoder, AttDecoder
from buildS2SDatasetForTest import AbstractiveDataset
from argparse import ArgumentParser

import torch.cuda as cutorch


parser = ArgumentParser()
parser.add_argument('--test_data_path', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

VAL_DATASET_PATH = args.test_data_path
OUTPUT_PATH = args.output_path

WORD_EMBEDDING_DATA = "./data/wordEmbedding.npy"
WORD2INDEX_PATH = "./data/word2index.json"
INDEX2WORD_PATH = "./data/index2word.json"

#MODEL = "./Attention/attention.model" 
MODEL = "./RL0.5.model" 

EMBEDDING_DIMENSION = 300
HIDDEN_DIMENSION = 250
LEARNING_RATE = 0.001
TRAINING_EPOCH = 30
BIDIRECTION = True
BATCH_SIZE = 64
RNN_LAYER = 1
DROPOUT = 0.00

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
	logging.info("Device type is '%s'" %(device))
	logging.info("Load %s"%(INDEX2WORD_PATH))
	with open(INDEX2WORD_PATH) as f:
		i2w = json.load(f)
	with open(WORD2INDEX_PATH) as f:
		w2i = json.load(f)
	logging.info("Load word embedding")
	word_embedding = np.load(WORD_EMBEDDING_DATA)
	logging.info("Read validation dataset")
	valSet = buildS2SDatasetForTest.build(VAL_DATASET_PATH)

	logging.info("Build data loader for validation data")
	validation_generator = data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=False, collate_fn=valSet.collate_fn)
	logging.info("Data loader loading is complete")

	loss_function = nn.CrossEntropyLoss(ignore_index=0)
	logging.info("Build encoder with hidden dimension %s" %(HIDDEN_DIMENSION))
	encoder = Encoder(EMBEDDING_DIMENSION, HIDDEN_DIMENSION, word_embedding.shape[0], word_embedding, RNN_LAYER, DROPOUT, BIDIRECTION) 
	logging.info("Build decoder with hidden dimension %s" %(HIDDEN_DIMENSION))
	decoder = AttDecoder(EMBEDDING_DIMENSION, HIDDEN_DIMENSION, word_embedding.shape[0], word_embedding, RNN_LAYER, DROPOUT) 
	logging.info("Build seq2seq model")
	model = Seq2Seq(encoder, decoder, device)
	model = torch.load(MODEL)
	#model = torch.load(MODEL, map_location=device)
	model.to(device)
	check_model_performance = -1
	#torch.set_printoptions(threshold=100000)
	model.eval()
	logging.info("Start validation")
	final =[]
	box = []
	ID = []
	with torch.no_grad():
		for step , d in enumerate(validation_generator):
			if step % 50 == 0: logging.info("Valid step %s" %(step))
			text = d['text'].to(device, dtype=torch.long)
			length = d['len_text']
			mask = d['attention_mask'].to(device, dtype=torch.long)
			out,predict = model.predict(text,1,2, length,mask)
			box.append(predict)
			ID.append(d['id'])
			del out, predict, text, mask
	for predict, test_idx in zip(box,ID):
		pre = 3
		for idx, ii in enumerate(predict):
			ans = []
			for s, j in enumerate(ii):
				if j == pre or i2w[j] == "<unk>": continue
				if i2w[j] == "</s>" or s > 80: break
				ans.append(i2w[j])
				pre = j
			sent = " ".join(ans)
			s_ans = {"id": test_idx[idx],"predict":sent}
			final.append(s_ans)
	logging.info("end predict")
	f = open(OUTPUT_PATH, "w")
	f.write("\n".join([json.dumps(p)for p in final])+"\n")
	
if __name__ == "__main__":
	loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
	logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
	main()
