import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

	def __init__(self, args, data, vectors):
		super(CNN, self).__init__()

		self.args = args

		self.word_emb = nn.Embedding(args.embed_num, args.embed_dim, padding_idx=1)
		# initialize word embedding with pretrained word2vec
		if args.mode != 'rand':
			self.word_emb.weight.data.copy_(torch.from_numpy(vectors))
		if args.mode in ('static', 'multichannel'):
			self.word_emb.weight.requires_grad = False
		if args.mode == 'multichannel':
			self.word_emb_multi = nn.Embedding(args.embed_num, args.embed_dim, padding_idx=1)
			self.word_emb_multi.weight.data.copy_(torch.from_numpy(vectors))
			self.in_channels = 2
		else:
			self.in_channels = 1

		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.word_emb.weight.data[0], -0.05, 0.05)

		for filter_size in args.kernel_sizes:
			conv = nn.Conv1d(self.in_channels, args.kernel_num, args.embed_dim * filter_size, stride=args.embed_dim)
			setattr(self, 'conv_' + str(filter_size), conv)

		self.fc = nn.Linear(len(args.kernel_sizes) * args.kernel_num, args.class_num)

	def forward(self, batch):
		x = torch.t(batch.text)
		#x = torch.t(batch)
		batch_size, seq_len = x.size()

		conv_in = self.word_emb(x).view(batch_size, 1, -1)
		if self.args.mode == 'multichannel':
			#xembed = self.word_emb_multi(x)
			conv_in_multi = self.word_emb_multi(x).view(batch_size, 1, -1)
			conv_in = torch.cat((conv_in, conv_in_multi), 1)

		conv_result = [
			F.max_pool1d(F.relu(getattr(self, 'conv_' + str(filter_size))(conv_in)), seq_len - filter_size + 1).view(-1,
																													self.args.kernel_num)
			for filter_size in self.args.kernel_sizes]

		out = torch.cat(conv_result, 1)
		out = F.dropout(out, p=self.args.dropout, training=self.training)
		out = self.fc(out)

		return out