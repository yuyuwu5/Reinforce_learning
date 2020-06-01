import os
import json
import torch
import pickle
import spacy
import torch
import logging
from tqdm import tqdm
from util import padding
from torch.utils.data import Dataset

WORD2INDEX_PATH = "./data/word2index.json"

class AbstractiveDataset(Dataset):
	ignore_idx = -100
	def __init__(self, data, padding=0, max_text_len=270, max_summary_len = 70):
		self.data = data
		self.padding = padding
		self.max_text_len = max_text_len
		self.max_summary_len = max_summary_len

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		sample = self.data[index]
		instance = {
				'id': sample['id'],
				'text': sample['text'][:self.max_text_len],
				'len_text': min(len(sample['text']),self.max_text_len),
				'attention_mask': [True]*min(len(sample['text']), self.max_text_len)
		}
		if 'summary' in sample:
			instance['summary'] = sample['summary'][:self.max_summary_len]
		return instance

	def collate_fn(self, samples):
		batch = {}
		for key in ['id', 'len_text']:
			if any (key not in sample for sample in samples): continue
			batch[key] = [sample[key] for sample in samples]
		for key in ['text', 'summary', 'attention_mask']:
			if any(key not in sample for sample in samples):continue
			to_len = max([len(sample[key]) for sample in samples])
			padd = padding([sample[key] for sample in samples], to_len, self.padding)
			batch[key] = torch.tensor(padd)
		return batch

def token2Id(token, dic):
	token = token.lower()
	return dic.get(token) or dic['<unk>']

def sent2num(words, dic, nlp):
	word = [token.text for token in nlp(words) if token.is_alpha or token.is_digit]
	return [token2Id(t, dic) for t in word]

def processS2SSample(data, dic):
	processed = []
	bos_id = dic['<s>']
	eos_id = dic['</s>']
	nlp = spacy.load('en_core_web_sm',disable=['tagger', 'ner', 'parser', 'textcat'])
	for sample in tqdm(data):
		p = {
				'id': sample['id'],
				'text': sent2num(sample['text'], dic, nlp)+[eos_id],
			}
		if 'summary' in sample:
			p['summary'] = ([bos_id] + sent2num(sample['summary'], dic, nlp)+[eos_id])
		processed.append(p)
	return processed

def buildS2SDataset(samples, padding):
	return AbstractiveDataset(samples, padding=padding, max_text_len=300, max_summary_len=80)

def build(path):
	logging.info('Open testing data')
	with open(path) as f:
		test = [json.loads(l) for l in f]
	logging.info('Open word 2 index')
	with open(WORD2INDEX_PATH) as f:
		dic = [json.loads(l) for l in f]
	dic = dic[0]
	logging.info('Build testing dataset')
	return buildS2SDataset(
			processS2SSample(test, dic),
			dic['<pad>'],
			)
