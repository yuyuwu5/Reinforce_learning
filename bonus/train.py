import os
import logging
import json
import pickle
import judge
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils import data
from model import Seq2Seq, Encoder, AttDecoder
from buildS2SDataset import AbstractiveDataset

TRAIN_DATASET_PATH = "./data/trainS2S.pkl"
VAL_DATASET_PATH = "./data/validS2S.pkl"
WORD_EMBEDDING_DATA = "./data/wordEmbedding.npy"
WORD2INDEX_PATH = "./data/word2index.json"
INDEX2WORD_PATH = "./data/index2word.json"


EMBEDDING_DIMENSION = 300
HIDDEN_DIMENSION = 128
LEARNING_RATE = 0.001
TRAINING_EPOCH = 20
BIDIRECTION = True


BATCH_SIZE = 10
RL_RATIO = 0.9

RNN_LAYER = 1
DROPOUT = 0.2
FORCE = 1


EVAL_PERFORMANCE = "./RL%s.txt"%(RL_RATIO) 
MODEL_PATH = "./RL%s.model"%(RL_RATIO) 

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def translate(pred, summary_ans, i2w, device):
	target_sents = []
	predict_sents =[]
	tar = ""
	summary = ""

	for idx, ii in enumerate(pred):
		ans = []
		pre = 3
		for s, j in enumerate(ii):
			if pre == j or i2w[j] == "<unk>": continue
			if i2w[j] == "</s>" or s > 80: break
			ans.append(i2w[j])
			pre = j
		sent = " ".join(ans)
		realans = summary_ans[idx]
		tar += sent + " "
		summary += realans + " "
		target_sents.append(realans)
		predict_sents.append(sent)
	result = judge.rougeL(target_sents, predict_sents)
	#result = judge.rougeL([tar], [summary])
	return result.to(device)


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info("Device type is '%s'" %(device))
	logging.info("Load %s"%(INDEX2WORD_PATH))
	with open(INDEX2WORD_PATH) as f:
		i2w = json.load(f)
	with open(WORD2INDEX_PATH) as f:
		w2i = json.load(f)
	logging.info("Load word embedding")
	word_embedding = np.load(WORD_EMBEDDING_DATA)
	logging.info("Read training dataset")
	f = open(TRAIN_DATASET_PATH ,"rb")
	trainSet = pickle.load(f)
	f.close()
	logging.info("Read validation dataset")
	f = open(VAL_DATASET_PATH, "rb")
	valSet = pickle.load(f)
	f.close()

	logging.info("Build data loader for training data")
	training_generator = data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, collate_fn=trainSet.collate_fn, num_workers = 4)
	logging.info("Data loader loading is complete")
	logging.info("Build data loader for validation data")
	validation_generator = data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=True, collate_fn=valSet.collate_fn, num_workers = 4)
	logging.info("Data loader loading is complete")

	loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
	logging.info("Build encoder with hidden dimension %s" %(HIDDEN_DIMENSION))
	encoder = Encoder(EMBEDDING_DIMENSION, HIDDEN_DIMENSION, word_embedding.shape[0], word_embedding, RNN_LAYER, DROPOUT, BIDIRECTION) 
	logging.info("Build decoder with hidden dimension %s" %(HIDDEN_DIMENSION))
	decoder = AttDecoder(EMBEDDING_DIMENSION, HIDDEN_DIMENSION, word_embedding.shape[0], word_embedding, RNN_LAYER, DROPOUT) 
	logging.info("Build seq2seq model")
	model = Seq2Seq(encoder, decoder, device)
	print(model)
	model.apply(init_weights)
	optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
	model.to(device)
	logging.info("Start Training")
	check_model_performance = -1
	torch.set_printoptions(threshold=100)
	for i in range(TRAINING_EPOCH):
		epoch_loss = 0
		#RL_RATIO = min((i//3) * 0.1, 1)
		for step, d in enumerate(training_generator):
			model.train()
			text = d['text']
			summary = d['summary']
			length = d['len_text']
			mask = d['attention_mask']
			text = text.to (device, dtype=torch.long)
			summary = summary.to(device, dtype=torch.long)
			mask = mask.to(device)
			output, pred, _ = model(text, summary, FORCE, length, mask, True)
			#_, baseline = model(text, summary, FORCE, length, mask, True)
			sampleOut, sample, _ = model(text, summary, FORCE, length, mask, False)
			print(sample)

			output = output[:, 1:].reshape(-1, output.size(2))
			sampleOut = sampleOut[:, 1:].reshape(-1, sampleOut.size(2))
			summary = summary[:, 1:].reshape(-1)
			loss_greedy = loss_function(output, summary).sum()
			loss_sample = loss_function(sampleOut, summary).reshape(mask.shape[0], -1).sum(1)
			
			### RL Part
			print("start rouge")
			baseline_rouge = translate(pred, d['summary_w'], i2w, device)
			print("end rouge")
			print("start rouge")
			sample_rouge = translate(sample, d['summary_w'], i2w, device)
			print("end rouge")

			loss = (1-RL_RATIO)*loss_greedy + RL_RATIO* ((baseline_rouge - sample_rouge) * loss_sample).sum()
			########
			optimizer.zero_grad()
			epoch_loss += loss.item()
			loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
			optimizer.step()
			if step % 100 == 0:
				logging.info("Epoch: %s step: %s, Ein=%s"%(i+1, step, epoch_loss/(step+1)))
		logging.info("Iter %s, overall performance %s" %(i+1, epoch_loss))
		model.eval()
		logging.info("Start validation")
		for datas in [validation_generator]:
			target_sents = []
			predict_sents =[]
			for step , d in enumerate(datas):
				if step % 200 == 0 and step > 0:
					logging.info("Valid step %s" %(step))
					break
				with torch.no_grad():
					text = d['text']
					length = d['len_text']
					mask = d['attention_mask']
					text = text.to(device, dtype=torch.long)
					mask = mask.to(device)
					output, predict = model.predict(text, w2i["<s>"], w2i["</s>"], length, mask)
					for idx, ii in enumerate(predict):
						ans = []
						pre = 3
						for s, j in enumerate(ii):
							if pre == j or i2w[j] == "<unk>": continue
							if i2w[j] == "</s>" or s > 80: break
							ans.append(i2w[j])
							pre = j
						sent = " ".join(ans)
						ans = d['summary_w'][idx]
						target_sents.append(ans)
						predict_sents.append(sent)
						#print(sent)
						#print(ans)
				#print(predict_sents[len(predict_sents)-1])
		logging.info("end predict")
		result = judge.extractiveJudge(target_sents, predict_sents)
		logging.info(result)
		f = open(EVAL_PERFORMANCE, "a")
		f.write("Iteration %s:\n" %(i+1))
		f.write(str(result))
		f.write("\n")
		f.close()
		m = (result['mean']['rouge-1']+result['mean']['rouge-2']+result['mean']['rouge-l'])/3
		logging.info("End validation")
		if m > check_model_performance:
			check_model_performance = m
			logging.info("Iter %s , save model, overall performance %s" %(i+1, check_model_performance))
			torch.save(model, MODEL_PATH)
if __name__ == "__main__":
	loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
	logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
	main()
