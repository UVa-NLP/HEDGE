import numpy as np
import itertools
from itertools import combinations
import torch
from copy import copy, deepcopy

class Shapley_TopDown_Tree:
    def __init__(self, model, batch, pad_index = 0, win_size = 5, device='cuda'):
        score = model(batch).detach().cpu().numpy()
        score_norm = (np.exp(score) / np.sum(np.exp(score), axis=1))
        self.pred_label = np.argmax(score_norm)
   
        self.output = []
        self.fea_num = len(batch.text)
        self.pad = pad_index
   
        self.model = model
        self.input = batch.text
        self.win_size = win_size
        self.device = torch.device(device)
        mask_input = torch.ones(self.input.shape, dtype=torch.long)*self.pad
        class temp:
            pass
        temp.text = mask_input.to(self.device)
        score = model(temp).detach().cpu().numpy()
        score_norm = (np.exp(score) / np.sum(np.exp(score), axis=1))
        self.bias = score_norm[0][self.pred_label]
        self.valdict = dict()


    def set_contribution_func(self, fea_set):
        # input has just one sentence, input is a list
        flat_fea = [f for fea in fea_set for f in fea]
        flat_fea = frozenset(flat_fea)
        if flat_fea in self.valdict:
            return self.valdict[frozenset(flat_fea)]
        input = self.input
        mask_input = torch.ones(input.shape, dtype=torch.long)*self.pad
        class temp:
            pass
        # mask the input with zero
        for fea_idx in fea_set:
                for idx in fea_idx:
                    mask_input[idx] = input[idx]
        temp.text = mask_input.to(self.device)
        # send the mask_input into model
        score = self.model(temp).detach().cpu().numpy()
        score_norm = (np.exp(score) / np.sum(np.exp(score), axis=1))
        contri = score_norm[0][self.pred_label] - self.bias
        self.valdict[frozenset(flat_fea)] = contri
        return contri



    def get_shapley_interaction_weight(self, d, s):
        return np.math.factorial(s) * np.math.factorial(d - s - 2) / np.math.factorial(d - 1) / 2

    def shapley_interaction_approx(self, feature_set, left, right):
        win_size = self.win_size
        if left + 1 != right:
            print("Not adjacent interaction")
            return -1
        fea_num = len(feature_set)
        curr_set_lr = [feature_set[left], feature_set[right]]
        curr_set_l = [feature_set[left]]
        curr_set_r = [feature_set[right]]
        if left + 1 - win_size > 0:
            left_set = feature_set[left - win_size:left]
        else:
            left_set = feature_set[0:left]
        if right + win_size > fea_num - 1:
            right_set = feature_set[right + 1:]
        else:
            right_set = feature_set[right + 1:right + win_size + 1]
        adj_set = left_set + right_set
        num_adj = len(adj_set)
        dict_subset = {r: list(combinations(adj_set, r)) for r in range(num_adj + 1)}
        score = 0.0
        for i in range(num_adj + 1):
            weight = self.get_shapley_interaction_weight(fea_num, i)
            if i == 0:
                score_included = self.set_contribution_func(curr_set_lr)
                score_excluded_l = self.set_contribution_func(curr_set_r)
                score_excluded_r = self.set_contribution_func(curr_set_l)
                score_excluded = 0
                score += (score_included - score_excluded_l - score_excluded_r + score_excluded) * weight
            else:
                for subsets in dict_subset[i]:
                    score_included = self.set_contribution_func(list(subsets) + curr_set_lr)
                    score_excluded_l = self.set_contribution_func(list(subsets) + curr_set_r)
                    score_excluded_r = self.set_contribution_func(list(subsets) + curr_set_l)
                    score_excluded = self.set_contribution_func(list(subsets))
                    score += (score_included - score_excluded_l - score_excluded_r + score_excluded) * weight
        return score

    def shapley_topdown_tree(self):
        input = self.input
        fea_num = len(input)
        if fea_num == 0:
            return -1
        fea_set = [list(range(fea_num))]
       
        pos = 0
        # level = 0
        hier_tree = {}
        hier_tree[0] = fea_set
        for level in range(1, self.fea_num):
            pos = 0
            min_inter_score = 1e8
            pos_opt = 0
            inter_idx_opt = 0
            while pos < len(fea_set):
                subset = fea_set[pos]
                sen_len = len(subset)
                if sen_len == 1:
                    pos += 1
                    continue
                new_fea_set = [ele for x, ele in enumerate(fea_set) if x != pos]
                score_buff = []
                for idx in range(1, sen_len):
                    leave_one_set = deepcopy(new_fea_set)
                    sub_set1 = subset[0:idx]
                    sub_set2 = subset[idx:]
                    leave_one_set.insert(pos, sub_set1)
                    leave_one_set.insert(pos + 1, sub_set2)
                    score_buff.append(self.shapley_interaction_approx(leave_one_set, pos, pos + 1))
                inter_score = np.array(score_buff)
                min_inter_idx = np.argmin(inter_score)
                minter = inter_score[min_inter_idx]
                if minter < min_inter_score:
                    min_inter_score = minter
                    inter_idx_opt = min_inter_idx
                    pos_opt = pos
                pos += 1

            new_fea_set = [ele for x, ele in enumerate(fea_set) if x != pos_opt]
            subset = fea_set[pos_opt]
            sub_set1 = subset[0:inter_idx_opt + 1]
            sub_set2 = subset[inter_idx_opt + 1:]
            new_fea_set.insert(pos_opt, sub_set1)
            new_fea_set.insert(pos_opt + 1, sub_set2)
            fea_set = new_fea_set
            hier_tree[level] = fea_set
        self.max_level = level
        self.hier_tree = hier_tree
        return hier_tree

    def compute_shapley_hier_tree(self):
        hier_tree = self.shapley_topdown_tree()
        self.hier_tree = {}
        for level in range(self.max_level+1):
            self.hier_tree[level] = []
            for idx, subset in enumerate(hier_tree[level]):
                score = 2*(self.set_contribution_func([subset])+self.bias)-1
                self.hier_tree[level].append((subset,score))
        return self.hier_tree



    def get_importance_phrase(self):
        phrase_dict = dict()
        for level in range(1,self.max_level+1):
            for fea_set, score in self.hier_tree[level]:
                phrase_dict[frozenset(fea_set)] = score
        phrase_tuple = sorted(phrase_dict.items(), key=lambda item: item[1], reverse=True)
        phrase_list = [list(item[0]) for item in phrase_tuple]
        score_list = [item[1] for item in phrase_tuple]
        return phrase_list, score_list


    def visualize_tree(self, batch, wordvocab, fontsize=8):
        levels = self.max_level
        vals = np.array([fea[1] for level in range(levels) for fea in self.hier_tree[level]])
        min_val = np.min(vals)
        max_val = np.max(vals)
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        # max_color = max_val if max_val>0 else -max_val+0.1
        # min_color = min_val if min_val<0 else -min_val-0.1
        max_color = 1
        min_color = -1
        cnorm = mpl.colors.Normalize(vmin=min_color, vmax=max_color, clip=False)
    
        if self.pred_label == 1: #1 stands for positive
            cmapper = mpl.cm.ScalarMappable(norm=cnorm, cmap='RdYlBu')
        else: #0 stands for negative
            cmapper = mpl.cm.ScalarMappable(norm=cnorm, cmap='RdYlBu_r')

        words = batch.text.numpy()
        nwords = words.shape[0]
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.xaxis.set_visible(False)

        ylabels = ['Step '+str(idx) for idx in range(self.max_level+1)]
        ax.set_yticks(list(range(0, self.max_level+1)))
        ax.set_yticklabels(ylabels,fontsize=18)
        ax.set_ylim(self.max_level+0.5, 0-0.5)

        sep_len = 0.3
        for key in range(levels+1):
            for fea in self.hier_tree[key]:
                len_fea = 1 if type(fea[0]) == int else len(fea[0])
                start_fea = fea[0] if type(fea[0])==int else fea[0][0]
                start = sep_len * start_fea + start_fea + 0.5
                width = len_fea + sep_len * (len_fea - 1)
                fea_color = cmapper.to_rgba(fea[1])
                r, g, b, _ = fea_color
                c = ax.barh(key, width=width, height=0.5, left=start, color=fea_color)
                text_color = 'white' if r * g * b < 0.2 else 'black'
                #         text_color = 'black'
                word_idxs = fea[0]
                for i, idx in enumerate(word_idxs):
                    word_pos = start + sep_len * (i) + i + 0.5
                    word_str = wordvocab[batch.text[idx]]
                    ax.text(word_pos, key, word_str, ha='center', va='center',
                            color=text_color, fontsize=fontsize)
                    word_pos += sep_len
                start += (width + sep_len)
        cb = fig.colorbar(cmapper, ax=ax)
        cb.ax.tick_params(labelsize=18)
        plt.show()
