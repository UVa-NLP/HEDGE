import torch
import numpy as np
import random
from tqdm import tqdm
class BATCH:
    pass

def form_a_batch(s, word_dic_full, device='cuda'):
    batch = BATCH()
    nums = np.expand_dims(np.array([word_dic_full[x] for x in s]).transpose(), axis=1)
    batch.text = torch.LongTensor(nums).to(device)
    return batch

def calc_cohesion(model, filename, res_file, max_len, word_dic_full, device, num=100, max_sen=2000):
    # read text
    textlist = []
    fealist = []
    delta_dict = {}
    with open(filename,'r') as fp:
        lines = fp.readlines()
    for ii in range(1, len(lines), 2):
        line = lines[ii].split(' ||| ')
        line1 = line[0].split(' >> ')
        line2 = line[1].split(' >> ')
        textlist.append(line1[0].split(' ')[:-1])
        fealist.append(line2[0].split(' ')[:-1])
    # calculate cohesion
    S_faith, count, avg_len = 0, 0, 0
    for txt, fea in tqdm(zip(textlist, fealist)):
        count += 1
        txt_ori = txt.copy()
        batch_txt_ori = form_a_batch(txt_ori, word_dic_full, device)
        print(batch_txt_ori)
        prob_txt_ori = model(batch_txt_ori)
        
        
        prob_txt_ori_norm = (np.exp(prob_txt_ori.detach().cpu().numpy()) / np.sum(np.exp(prob_txt_ori.detach().cpu().numpy()), axis=1))
        prob_ori = prob_txt_ori_norm.max()
        pred_ori = prob_txt_ori_norm.argmax()
        phrase_idxes = []

        for phrase in fea:
            indexes = phrase.split('-')
            start = int(phrase[0])
            if len(indexes) == 1:
                end = start
            else:
                end = int(indexes[1])
            phrase_idxes = list(range(start, end + 1))
            if max_len == -1:
                break
            else:
                if len(phrase_idxes) <= max_len:
                    break
        avg_len += len(phrase_idxes)
        delta = avg_phrase_importance(model, batch_txt_ori, phrase_idxes, pred_ori, prob_ori,device, num)
        res_file.write(str(delta)+' '+str(prob_ori)+' '+str(len(phrase_idxes)))
        res_file.write('\n')
        S_faith += delta
        if count == max_sen:
            break
    S_faith /= count
    avg_len /= count
    print('\nCohesion score for {} is {:.6f}, avg_len={}'.format(filename, S_faith, avg_len))
    return S_faith

def avg_phrase_importance(model, batch, phrase_idxes, label, prob, device='cuda', num=100):
    input_text = batch.text.cpu()
    sen_len = len(input_text)

    #generate (word, pos) pairs

    np.random.seed(0)
    start = phrase_idxes[0]
    end = phrase_idxes[-1]
    phrase_len = len(phrase_idxes)
    delta_sum = 0
    count = 0
    while count<num:
        word_pos_arr = [[i, i] for i in range(sen_len)]
        pos_arr = np.random.choice(np.arange(-phrase_len,sen_len+phrase_len), phrase_len, False)
        for i, key in enumerate(range(start,end+1)):
            word_pos_arr[key][1] = pos_arr[i]
        word_pos_arr = sorted(word_pos_arr,key=lambda item:item[1])
        shuffled_texts = [input_text[word] for word, pos in word_pos_arr]
        nums = np.expand_dims(np.array(shuffled_texts).transpose(), axis=1)
        batch.text = torch.LongTensor(nums).to(device)  # cuda()
        prob_txt_shuffle = model(batch)
        prob_txt_shuffle_norm = (
                    np.exp(prob_txt_shuffle.detach().cpu().numpy()) / np.sum(np.exp(prob_txt_shuffle.detach().cpu().numpy()), axis=1))
        delta = prob - prob_txt_shuffle_norm[0][label]

        delta_sum += delta
        count += 1
    return delta_sum/num
