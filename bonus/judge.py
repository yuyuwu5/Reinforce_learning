import logging
import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from rouge_score.rouge_scorer import RougeScorer

ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']
USE_STEMMER = False

def extractiveJudge(target, predict):
	rouge_score = RougeScorer(ROUGE_TYPES, use_stemmer=USE_STEMMER)
	with Pool(cpu_count()) as pool:
		scores = pool.starmap(rouge_score.score, [(t, p) for t, p in zip(target, predict)])
	r1s = np.array([s['rouge1'].fmeasure for s in scores])
	r2s = np.array([s['rouge2'].fmeasure for s in scores])
	rls = np.array([s['rougeL'].fmeasure for s in scores])
	scores = {
			'mean': {
				'rouge-1': r1s.mean(),
				'rouge-2': r2s.mean(),
				'rouge-l': rls.mean()
				},
			'std': {
				'rouge-1': r1s.std(),
				'rouge-2': r2s.std(),
				'rouge-l': rls.std()
				}
			}
	#logging.info("Evaluation score is\n%s" %(scores))
	return scores
def rougeL(target, predict):
	rouge_score = RougeScorer(ROUGE_TYPES, use_stemmer=USE_STEMMER)
	with Pool(cpu_count()) as pool:
		scores = pool.starmap(rouge_score.score, [(t, p) for t, p in zip(target, predict)])
	rls = torch.tensor([s['rougeL'].fmeasure for s in scores])
	return rls
