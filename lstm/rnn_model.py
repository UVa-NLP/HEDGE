import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init

class LSTM(nn.Module):

	def __init__(self, args, data, vectors):
		super(LSTM, self).__init__()

		self.args = args

		self.embed = nn.Embedding(args.embed_num, args.embed_dim, padding_idx=1)
		# initialize word embedding with pretrained word2vec
		self.embed.weight.data.copy_(torch.from_numpy(vectors))

		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.embed.weight.data[0], -0.05, 0.05)

		# lstm
		self.lstm = nn.LSTM(args.embed_dim, args.hidden_dim, dropout=args.dropout, num_layers=args.hidden_layer)
		# initial weight
		init.xavier_normal(self.lstm.all_weights[0][0], gain=np.sqrt(6.0))
		init.xavier_normal(self.lstm.all_weights[0][1], gain=np.sqrt(6.0))

		# linear
		self.hidden2label = nn.Linear(args.hidden_dim, args.class_num)
		# dropout
		self.dropout = nn.Dropout(args.dropout)
		self.dropout_embed = nn.Dropout(args.dropout)

	def forward(self, batch):
		x = batch.text
		embed = self.embed(x)
		embed = self.dropout_embed(embed)
		x = embed.view(len(x), embed.size(1), -1)
		# lstm
		lstm_out, _ = self.lstm(x)
		# lstm_out, self.hidden = self.lstm(x, self.hidden)
		lstm_out = torch.transpose(lstm_out, 0, 1)
		lstm_out = torch.transpose(lstm_out, 1, 2)
		# pooling
		lstm_out = F.tanh(lstm_out)
		lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
		lstm_out = F.tanh(lstm_out)
		# linear
		logit = self.hidden2label(lstm_out)
		return logit
