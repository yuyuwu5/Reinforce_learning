import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import CosineSimilarity

class Encoder(nn.Module):
	def __init__(self, embedding_dimension, hidden_dimension, dictionary_size, weight_matrix, layers, drop, bidirection):
		super(Encoder, self).__init__()
		self.word_embedding = nn.Embedding(dictionary_size, embedding_dimension)
		self.word_embedding.weight.data.copy_(torch.from_numpy(weight_matrix)) 
		self.hidden_dimension = hidden_dimension
		self.layers = layers
		self.gru = nn.GRU(embedding_dimension, hidden_dimension, num_layers=layers, batch_first=True, bidirectional=True)
		self.l = nn.Linear(hidden_dimension*2, hidden_dimension)
		self.dropout = nn.Dropout(drop)

	def forward(self, sentence, length):
		embedded = self.dropout(self.word_embedding(sentence))
		embedded = pack_padded_sequence(embedded, torch.LongTensor(length), batch_first=True, enforce_sorted=False)
		output, hidden = self.gru(embedded)
		output, _ = pad_packed_sequence(output, batch_first=True)
		hidden = torch.tanh(self.l(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
		return output, hidden

class AttDecoder(nn.Module):
	def __init__(self, embedding_dimension, hidden_dimension, dictionary_size, weight_matrix, layers, drop):
		super(AttDecoder, self).__init__()
		self.vocab_size = dictionary_size
		self.attention = Attention(hidden_dimension)
		self.word_embedding = nn.Embedding(dictionary_size, embedding_dimension)
		self.word_embedding.weight.data.copy_(torch.from_numpy(weight_matrix)) 
		self.gru = nn.GRU(embedding_dimension, hidden_dimension, num_layers=layers, batch_first=True, bidirectional=False)
		self.l1 = nn.Linear(hidden_dimension*3, dictionary_size)
		self.dropout = nn.Dropout(drop)

	def forward(self, sent, hidden_dimension, encoder_out, mask):
		sent = sent.unsqueeze(1)
		embedded_output = self.dropout(self.word_embedding(sent))
		rnnin = embedded_output
		output, hidden = self.gru(rnnin, hidden_dimension.unsqueeze(0))
		hidden = hidden.squeeze(0)
		attention_weight = self.attention(hidden, encoder_out, mask)
		attention_weight = attention_weight.unsqueeze(1)
		context = torch.bmm(attention_weight,encoder_out)
		#rnnin = torch.cat([embedded_output, context], 2)
		predict = self.l1(torch.cat([output.squeeze(1), context.squeeze(1)],1))
		return predict, hidden.squeeze(0),attention_weight

class Seq2Seq(nn.Module):
	def __init__(self, encoder, decoder, device):
		super().__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.device = device
	def forward(self, sent, target, power_force_ratio,length,mask, greedy):
		batch_size = target.shape[0]
		target_len = target.shape[1]
		vocab_size = self.decoder.vocab_size
		outputs = torch.zeros([batch_size, target_len, vocab_size]).to(self.device)
		encoder_out, hidden = self.encoder(sent, length)
		input = target[:, 0]
		predict = []
		log_probability = []
		for t in range(1, target_len):
			output, hidden, _ = self.decoder(input, hidden, encoder_out,mask)
			outputs[:, t] = output
			force = random.random() <= power_force_ratio

			if greedy:
				top = output.argmax(1)
			else:
				'''
				out = torch.exp(output)
				prob = torch.distributions.Categorical(out)
				top = prob.sample()
				log_probability.append(prob.log_prob(top).unsqueeze(1))
				'''
				out = torch.exp(output)
				top = torch.multinomial(out, 1)
				top = top.squeeze(1)
			input = target[:, t] if force and t < target_len else top
			predict.append(top.unsqueeze(1))
		predict = torch.cat(predict, 1)
		log_prob = []

		return outputs, predict, log_prob

	def predict(self, sent, bos, eos, length, mask):
		batch_size = sent.shape[0]
		input_len = min(sent.shape[1], 70)
		vocab_size = self.decoder.vocab_size
		outputs = torch.zeros([batch_size, input_len, vocab_size]).to(self.device)
		attentions = torch.zeros(32, 70, input_len).to(self.device)
		input = torch.ones([batch_size], dtype=torch.long).to(self.device)
		encoder_out, hidden = self.encoder(sent, length)
		predict = []
		for t in range(input_len):
			output, hidden, _ = self.decoder(input, hidden, encoder_out, mask)
			outputs[:, t] = output
			top = output.argmax(1)
			input = top
			predict.append(top.unsqueeze(1))
		predict = torch.cat(predict,1)
		return outputs, predict

class Attention(nn.Module):
	def __init__(self, hidden_dimension):
		super(Attention, self).__init__()
		self.hidden_dimension = hidden_dimension
		self.attn = nn.Linear(hidden_dimension*3, hidden_dimension)
		self.v = nn.Linear(hidden_dimension, 1, bias=False)
	def forward(self, decoder_hide, encoder_out, mask):
		step = encoder_out.size(1)
		h = decoder_hide.unsqueeze(1).repeat(1,step,1)
		energy = torch.tanh(self.attn(torch.cat((h, encoder_out), dim=2)))
		attention = self.v(energy).squeeze(2)
		#cos = CosineSimilarity(dim=2, eps=1e-7)
		#attention = cos(h, encoder_out)
		attention = attention.masked_fill(mask == 0, -1e5)
		return F.softmax(attention, dim=1)
