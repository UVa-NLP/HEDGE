import numpy as np
from torchtext import data
from torchtext import datasets
from gensim.models import KeyedVectors

def getVectors(args, data):
    vectors = []
    if args.mode != 'rand':
        word2vec = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        for i in range(len(data.TEXT.vocab)):
            word = data.TEXT.vocab.itos[i]
            if word in word2vec.vocab:
                vectors.append(word2vec[word])
            else:
                vectors.append(np.random.uniform(-0.01, 0.01, args.embed_dim))
    else:
        for i in range(len(data.TEXT.vocab)):
            vectors.append(np.random.uniform(-0.01, 0.01, args.embed_dim))
    return np.array(vectors)


class DATA():

    def __init__(self, args, tokenizer):
        #self.TEXT = data.Field(tokenize = tokenizer, batch_first=True, lower=True, fix_length=70)
        self.TEXT = data.Field(tokenize=tokenizer, lower=True)
        self.LABEL = data.Field(sequential=False, unk_token=None)
        self.train, self.test = datasets.IMDB.splits(self.TEXT, self.LABEL)
        self.TEXT.build_vocab(self.train, self.test)
        #self.train_iter, self.dev_iter, self.test_iter = data.BucketIterator.splits((self.train, self.dev, self.test), batch_size=args.batch_size)
        self.train_iter, self.test_iter = data.Iterator.splits((self.train, self.test), batch_size=args.batch_size, sort=True, device=args.device)
        self.LABEL.build_vocab(self.train)