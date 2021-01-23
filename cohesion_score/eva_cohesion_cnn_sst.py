import argparse
import torch
from load_data import DATA
import os
import re
import numpy as np
import random
from tqdm import tqdm
import sys
sys.path.append('..')
import eval_utils as eval

os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser(description='CNN text classificer')
parser.add_argument('-lr', type=float, default=0.00003, help='initial learning rate')
parser.add_argument('--clip', type=float, default=1, help='gradient clipping')
parser.add_argument('--minseqlen', type=float, default=6, help='minimum sequence length')
parser.add_argument('--kwordsnum', type=float, default=3, help='number of key words')
parser.add_argument('-epochs', type=int, default=30, help='number of epochs for training')
parser.add_argument('-batch-size', type=int, default=1, help='batch size for training')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3, 4, 5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
parser.add_argument('--mode', type=str, default='multichannel',
                    help='available models: rand, static, non-static, multichannel')
parser.add_argument('--gpu', default=-1, type=int, help='0:gpu, -1:cpu')
parser.add_argument('--gpu_id', default='3', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

torch.manual_seed(args.seed)

if args.gpu > -1:
    args.device = "cuda"
else:
    args.device = "cpu"

def tokenizer(s):
    s_clean = string_clean(s)
    return s_clean.split()

def string_clean(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string

# load data
data = DATA(args, tokenizer)
train_iter = data.train_iter
test_iter = data.test_iter

# vocab
wordvocab = data.TEXT.vocab.itos

# full vocab
word_dic_full = {}
word_invdic_full = {}
for ii, ww in enumerate(wordvocab):
    word_dic_full[ww] = ii
    word_invdic_full[ii] = ww

args.embed_num = len(data.TEXT.vocab)
args.class_num = len(data.LABEL.vocab)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# load model
if args.gpu > -1:
    with open('model.pt', 'rb') as f:
        model = torch.load(f)
    model.to(torch.device(args.device))
else:
    with open('model.pt', 'rb') as f:
        model = torch.load(f, map_location='cpu')


if __name__ == '__main__':
    max_len = -1
    score_file = 'cohesion_shapley_cnn_sst_top_feature.txt'
    phrase_file = 'Shapley_topdown_cnn_sst.txt'
    with open(score_file, 'w') as f:
        eval.calc_cohesion(model, phrase_file, f, max_len, word_dic_full, args.device, num=100)
