import argparse
import torch
from load_data import DATA
import os
import re
import numpy as np
import itertools
import hedge as sptd
from copy import copy, deepcopy
import time

os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser(description='CNN text classificer')
parser.add_argument('-lr', type=float, default=0.00003, help='initial learning rate')
parser.add_argument('--clip', type=float, default=1, help='gradient clipping')
parser.add_argument('--minseqlen', type=float, default=6, help='minimum sequence length')
parser.add_argument('--kwordsnum', type=float, default=3, help='number of key words')
parser.add_argument('-epochs', type=int, default=30, help='number of epochs for training')
parser.add_argument('-batch-size', type=int, default=1, help='batch size for training')
parser.add_argument('-dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension')
parser.add_argument('-kernel-num', type=int, default=300, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3, 4, 5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
parser.add_argument('--mode', type=str, default='multichannel',
                    help='available models: rand, static, non-static, multichannel')
parser.add_argument('--gpu', default=-1, type=int, help='0:gpu, -1:cpu')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--visualize', type=int, default=-1, help='index of the sentence to visualize, set to -1 to generate interpretations for all the sentences')
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
    #string = re.sub(r"[^A-Za-z0-9\'\`]", " ", string)
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
#dev_iter = data.dev_iter
test_iter = data.test_iter
#vectors = getVectors(args, data)

# vocab
wordvocab = data.TEXT.vocab.itos

# full vocab
word_dic_full = {}
word_invdic_full = {}
for ii, ww in enumerate(wordvocab):
    word_dic_full[ww] = ii
    word_invdic_full[ii] = ww
pad_index = word_dic_full['<pad>']

args.embed_num = len(data.TEXT.vocab)
args.class_num = len(data.LABEL.vocab)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# load model
if args.gpu > -1:
    with open('model.pt', 'rb') as f:
        # model = torch.load(f, map_location='cpu')
        model = torch.load(f)
    model.to(torch.device(args.device))
else:
    with open('model.pt', 'rb') as f:
        model = torch.load(f, map_location='cpu')


# calculate accuracy and interpretation of test data
def cal_acc_inter(model, data_iter, fileobject, win_size=2, vis=-1, device='cuda'):
    acc, count = 0, 0
    if vis > -1:
        sen_iter = itertools.islice(data_iter, vis, vis+1)
    else:
        sen_iter = data_iter
    for batch in sen_iter:
        count += 1
        print(count,len(batch.text))
        fileobject.write(str(count))
        fileobject.write('\n')
        batlen = batch.text.size(0)
        if batlen < args.minseqlen:
            batchtempt = torch.ones(args.minseqlen, 1, dtype=torch.int64)
            batchtempt[0:batlen] = batch.text
            batch.text = batchtempt.to(torch.device(device))
        pred = model(batch)
        _, pred = pred.max(dim=1)
        acc += (pred.cpu().numpy() == batch.label.cpu().numpy()).sum()

        batchtxt = batch.text
        for btxt in batchtxt:
            if (wordvocab[btxt] != '<pad>' and wordvocab[btxt] != '<unk>'):
                fileobject.write(wordvocab[btxt])
                fileobject.write(' ')
        fileobject.write(' >> ')
        if str(batch.label.cpu().numpy()) == '[0]':
            fileobject.write('0')
            fileobject.write(' ||| ')
        else:
            fileobject.write('1')
            fileobject.write(' ||| ')

        shap = sptd.Shapley_TopDown_Tree(model, batch, pad_index=pad_index, win_size=win_size,device=device)
        shap.compute_shapley_hier_tree()
        word_list, _ = shap.get_importance_phrase()
        
        for feaidx in word_list:
            if len(feaidx) == 1:
                if (wordvocab[batchtxt[feaidx[0]]] != '<pad>' and wordvocab[batchtxt[feaidx[0]]] != '<unk>'):
                    fileobject.write(str(feaidx[0]))
                    fileobject.write(' ')
            else:
                fea_end = -1
                for fea in feaidx[-1::-1]:
                    if(wordvocab[batchtxt[fea]] != '<pad>' and wordvocab[batchtxt[fea]] != '<unk>'):
                        fea_end = fea
                        break
                if fea_end > -1 and fea_end>feaidx[0]:
                    fileobject.write(str(feaidx[0]))
                    fileobject.write('-')
                    fileobject.write(str(fea_end))
                    fileobject.write(' ')
        

        fileobject.write(' >> ')
        if str(pred.cpu().numpy()) == '[0]':
            fileobject.write('0')
        else:
            fileobject.write('1')
        fileobject.write('\n')

        if vis > -1:
            shap.visualize_tree(batch, wordvocab, fontsize=8, tag=vis)
    acc /= count
    return acc


if __name__ == '__main__':
    # test_data
    start_time = time.time()
    with open('hedge_interpretation_index.txt', 'w') as fileobject:
        test_acc = cal_acc_inter(model, test_iter, fileobject, win_size=2,vis=args.visualize, device=args.device)
        print('\ntest_acc {:.6f}'.format(test_acc))
    print('\nElapsed Time is {:.6f}'.format(time.time()-start_time))

