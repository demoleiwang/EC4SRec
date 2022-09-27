# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/9, 2020/9/29, 2021/7/15
# @Author : Yupeng Hou, Yushuo Chen, Xingyu Pan
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, xy_pan@foxmail.com

"""
recbole.data.dataloader.general_dataloader
################################################
"""

import numpy as np
import torch

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader, NegSampleDataLoader
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import InputType, ModelType, set_color

import math
import random

from scipy.special import softmax
import os
from tqdm import tqdm

class TrainDataLoader(NegSampleDataLoader):
    """:class:`TrainDataLoader` is a dataloader for training.
    It can generate negative interaction when :attr:`training_neg_sample_num` is not zero.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self._set_neg_sample_args(config, dataset, config['MODEL_INPUT_TYPE'], config['train_neg_sample_args'])
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        if config['method'] in ['DuoRec', 'DuoRec_XAUG']:
            self.same_target_index = dataset.same_target_index
            self.static_item_id_list = dataset['item_id_list'].detach().clone()
            self.static_item_length = dataset['item_length'].detach().clone()
            self.mapping_index = torch.arange(len(self.static_item_id_list))
        if config['method'] in ['DuoRec_XAUG']:
            if not os.path.exists("./saved/temp/"):
                os.makedirs("./saved/temp/")
            inter_sect_dir_file = './saved/temp/' + self.config['dataset'] + '_intersect.npy'
            union_sect_dir_file = './saved/temp/' + self.config['dataset'] + '_union.npy'
            if os.path.exists(inter_sect_dir_file):  # and self.config['dataset'] != 'ml-1m':
                self.intersect_target_index = np.load(inter_sect_dir_file, allow_pickle=True)
                self.union_target_index = np.load(union_sect_dir_file, allow_pickle=True)
            else: # this is for speeding up
                union_target_index = []
                intersect_target_index = []
                target_items = dataset.inter_feat['item_id']
                iter_data = tqdm(
                    self.same_target_index,
                    total=len(self.same_target_index),
                    ncols=100,
                    desc=set_color(f"Prepro   ", 'yellow'), )
                for target_index, targets in enumerate(iter_data):
                    o_seq_x = self.static_item_id_list[target_index]
                    o_seq_x = o_seq_x[:self.static_item_length[target_index]]
                    cand_seqs = self.static_item_id_list[targets]
                    cand_lengths = self.static_item_length[targets]
                    cand_seqs = cand_seqs.cpu().numpy()
                    cand_lengths = cand_lengths.cpu().numpy()
                    useful_seq = np.arange(cand_seqs.shape[1])
                    s_ids_target_index = []
                    ovelap_items_target_index = []
                    for cand_i in range(len(cand_seqs)):
                        cand_seq = cand_seqs[cand_i]
                        cand_length = cand_lengths[cand_i]
                        cand_seq_x = cand_seq[:cand_length]
                        s_ids = useful_seq[:cand_length][np.in1d(cand_seq_x, o_seq_x, assume_unique=True)]
                        ovelap_items = len(s_ids) * 1.0 / max(len(cand_seq_x),
                                                              len(o_seq_x))  # np.union1d(cand_seq_x, o_seq_x)

                        s_ids_target_index.append(s_ids)
                        ovelap_items_target_index.append(ovelap_items)

                    intersect_target_index.append(s_ids_target_index)
                    union_target_index.append(ovelap_items_target_index)
                np.save(inter_sect_dir_file, np.array(intersect_target_index))
                np.save(union_sect_dir_file, np.array(union_target_index))
                self.intersect_target_index = np.array(intersect_target_index)
                self.union_target_index = np.array(union_target_index)

    def _init_batch_size_and_step(self):
        batch_size = self.config['train_batch_size']
        if self.neg_sample_args['strategy'] == 'by':
            batch_num = max(batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        self._set_neg_sample_args(config, self.dataset, config['MODEL_INPUT_TYPE'], config['train_neg_sample_args'])
        super().update_config(config)

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()
        if self.config['method'] in ['DuoRec', 'DuoRec_XAUG']:
            self.same_target_index = self.same_target_index[self.dataset.inter_feat.index]
            self.mapping_index = self.mapping_index[self.dataset.inter_feat.index]
        if self.config['method'] in ['DuoRec_XAUG']:
            self.union_target_index = self.union_target_index[self.dataset.inter_feat.index]
            self.intersect_target_index = self.intersect_target_index[self.dataset.inter_feat.index]

    def update_start(self, start_flag):
        self.start_flag = start_flag

    def _next_batch_data(self):
        cur_data = self._neg_sampling(self.dataset[self.pr:self.pr + self.step])

        if self.config['method'] == 'CL4SRec':
            self.cl4srec_aug(cur_data)
        if self.config['method'] == 'CL4SRec_XAUG':
            if self.start_flag == 0:
                pass
                # self.cl4srec_aug(cur_data)
            else:
                self.cl4srec_aug_x(cur_data)
        if self.config['method'] == 'DuoRec':
            self.duorec_aug(cur_data, slice(self.pr, self.pr + self.step))
        if self.config['method'] == 'DuoRec_XAUG':
            if self.start_flag == 0:
                self.duorec_aug(cur_data, slice(self.pr, self.pr + self.step))
            else:
                self.duoxaiselrec_aug(cur_data, slice(self.pr, self.pr + self.step))

        self.pr += self.step
        return cur_data

    ### CL4SRec
    def cl4srec_aug(self, cur_data):
        def item_crop(seq, length, eta=0.9):
            num_left = math.floor(length * eta)
            crop_begin = random.randint(0, length - num_left)
            croped_item_seq = np.zeros(seq.shape[0])
            if crop_begin + num_left < seq.shape[0]:
                croped_item_seq[:num_left] = seq[crop_begin:crop_begin + num_left]
            else:
                croped_item_seq[:num_left] = seq[crop_begin:]
            return torch.tensor(croped_item_seq, dtype=torch.long), torch.tensor(num_left, dtype=torch.long)

        def item_mask(seq, length, gamma=0.1):
            num_mask = math.floor(length * gamma)
            mask_index = random.sample(range(length), k=num_mask)
            masked_item_seq = seq[:]
            masked_item_seq[mask_index] = 0  # self.dataset.item_num  # token 0 has been used for semantic masking
            return masked_item_seq, length

        def item_reorder(seq, length, beta=0.1):
            num_reorder = math.floor(length * beta)
            reorder_begin = random.randint(0, length - num_reorder)
            reordered_item_seq = seq[:]
            shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
            random.shuffle(shuffle_index)
            reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
            return reordered_item_seq, length

        seqs = cur_data['item_id_list'].clone()
        lengths = cur_data['item_length'].clone()

        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []
        for seq, length in zip(seqs, lengths):
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]
                aug_seq = seq
                aug_len = length
            if switch[0] == 0:
                aug_seq, aug_len = item_crop(seq.clone(), length.clone(), eta=self.config['eta'])
            elif switch[0] == 1:
                aug_seq, aug_len = item_mask(seq.clone(), length.clone(), gamma=self.config['gamma'])
            elif switch[0] == 2:
                aug_seq, aug_len = item_reorder(seq.clone(), length.clone(), beta=self.config['beta'])

            aug_seq1.append(aug_seq)
            aug_len1.append(aug_len)

            if switch[1] == 0:
                aug_seq, aug_len = item_crop(seq.clone(), length.clone(), eta=self.config['eta'])
            elif switch[1] == 1:
                aug_seq, aug_len = item_mask(seq.clone(), length.clone(), gamma=self.config['gamma'])
            elif switch[1] == 2:
                aug_seq, aug_len = item_reorder(seq.clone(), length.clone(), beta=self.config['beta'])

            aug_seq2.append(aug_seq)
            aug_len2.append(aug_len)

        cur_data.update(Interaction({'aug1': torch.stack(aug_seq1), 'aug_len1': torch.stack(aug_len1),
                                     'aug2': torch.stack(aug_seq2), 'aug_len2': torch.stack(aug_len2)}))

    def cl4srec_aug_x(self, cur_data):
        def item_crop(seq, length, attribution_score, eta=0.6):
            num_left = math.floor(length * eta)
            if num_left < 1:
                num_left = 1

            sorted_attribution_index = np.argsort(attribution_score[:length])
            crop_ids = list(sorted_attribution_index[-num_left:].numpy())
            croped_item_seq = np.zeros(seq.shape[0])
            croped_item_seq[:num_left] = seq[crop_ids]


            return torch.tensor(croped_item_seq, dtype=torch.long), torch.tensor(num_left, dtype=torch.long)

        def item_crop_neg(seq, length, attribution_score, eta=0.6):
            num_left = math.floor(length * eta)
            if num_left < 1:
                num_left = 1
            sorted_attribution_index = np.argsort(attribution_score[:length])
            crop_ids = list(sorted_attribution_index[:num_left].numpy())
            croped_item_seq = np.zeros(seq.shape[0])
            croped_item_seq[:num_left] = seq[crop_ids]
            return torch.tensor(croped_item_seq, dtype=torch.long), torch.tensor(num_left, dtype=torch.long)

        def item_mask(seq, length, attribution_score, gamma=0.3):
            num_mask = math.floor(length * (1 - gamma))
            if num_mask < 1:
                num_mask = 1

            sorted_attribution_index = np.argsort(attribution_score[:length]).numpy()
            mask_index = list(sorted_attribution_index[
                              :num_mask])  # random.sample(list(sorted_attribution_index[:int(length / 2)].numpy()), k=num_mask)

            masked_item_seq = seq[:]
            masked_item_seq[mask_index] = 0  # self.dataset.item_num  # token 0 has been used for semantic masking
            return masked_item_seq, length

        def item_mask_neg(seq, length, attribution_score, gamma=0.3):
            num_mask = math.floor(length * (1 - gamma))
            if num_mask < 1:
                num_mask = 1
            sorted_attribution_index = np.argsort(attribution_score[:length]).numpy()
            mask_index = list(sorted_attribution_index[
                              -num_mask:])  # random.sample(list(sorted_attribution_index[int(length / 2):].numpy()), k=num_mask)
            masked_item_seq = seq[:]
            masked_item_seq[mask_index] = 0  # self.dataset.item_num  # token 0 has been used for semantic masking
            return masked_item_seq, length

        def item_reorder(seq, length, attribution_score, beta=0.6):
            num_reorder = math.floor(length * (1 - beta))
            if num_reorder < 1:
                num_reorder = 1

            sorted_attribution_index = np.argsort(attribution_score[:length])
            reorder_ids = list(sorted_attribution_index[:num_reorder].numpy())
            reorder_shuffle = list(range(len(reorder_ids)))
            random.shuffle(reorder_shuffle)
            sub_seq = seq[reorder_ids][reorder_shuffle]
            seq[reorder_ids] = sub_seq

            return seq, length

        def item_reorder_neg(seq, length, attribution_score, beta=0.6):
            num_reorder = math.floor(length * (1 - beta))
            if num_reorder < 1:
                num_reorder = 1
            sorted_attribution_index = np.argsort(attribution_score[:length])
            reorder_ids = list(sorted_attribution_index[-num_reorder:].numpy())
            reorder_shuffle = list(range(len(reorder_ids)))
            random.shuffle(reorder_shuffle)
            sub_seq = seq[reorder_ids][reorder_shuffle]
            seq[reorder_ids] = sub_seq
            return seq, length

        seqs = cur_data['item_id_list'].clone()
        lengths = cur_data['item_length'].clone()
        # if self.clxai4srec_aug_type == 'popularity':
        #     attribution_scores = cur_data['popularity_attribution_scores'].clone()
        # elif self.clxai4srec_aug_type in ['saliency', 'ig', 'occlusion']:
        #     attribution_scores = cur_data['xai_attribution_scores'].clone()
        attribution_scores = cur_data['attribution_scores'].clone()

        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []
        neg_seq_ = []
        neg_seq_len_ = []
        for seq, length, attribution_score in zip(seqs, lengths, attribution_scores):
            if length > 1:
                switch = random.sample(range(3), k=2)
                neg_switch = random.sample(range(3), k=1)

            else:
                switch = [3, 3]
                aug_seq = seq
                aug_len = length

                neg_switch = [3]
                neg_seq = torch.zeros(seq.size(0), dtype=int)
                random_value = min(1, int(random.random() * (self.dataset.item_num - 1)))
                neg_seq[0] = random_value
                neg_seq_len = length

            if switch[0] == 0:
                aug_seq, aug_len = item_crop(seq.clone(), length.clone(), attribution_score, self.config['pos_r'])
                # aug_seq, aug_len = item_crop(seq.clone(), length.clone(), attribution_score, 0.3)
            elif switch[0] == 1:
                aug_seq, aug_len = item_mask(seq.clone(), length.clone(), attribution_score, self.config['pos_r'])
                # aug_seq, aug_len = item_mask(seq.clone(), length.clone(), attribution_score, 0.5)
            elif switch[0] == 2:
                aug_seq, aug_len = item_reorder(seq.clone(), length.clone(), attribution_score,
                                                self.config['pos_r'])
                # aug_seq, aug_len = item_reorder(seq.clone(), length.clone(), attribution_score, 0.7)

            aug_seq1.append(aug_seq)
            aug_len1.append(aug_len)

            if switch[1] == 0:
                aug_seq, aug_len = item_crop(seq.clone(), length.clone(), attribution_score, self.config['pos_r'])
                # aug_seq, aug_len = item_crop(seq.clone(), length.clone(), attribution_score, 0.3)
            elif switch[1] == 1:
                aug_seq, aug_len = item_mask(seq.clone(), length.clone(), attribution_score, self.config['pos_r'])
                # aug_seq, aug_len = item_mask(seq.clone(), length.clone(), attribution_score, 0.5)
            elif switch[1] == 2:
                aug_seq, aug_len = item_reorder(seq.clone(), length.clone(), attribution_score,
                                                self.config['pos_r'])
                # aug_seq, aug_len = item_reorder(seq.clone(), length.clone(), attribution_score, 0.7)

            aug_seq2.append(aug_seq)
            aug_len2.append(aug_len)

            if neg_switch[0] == 0:
                neg_seq, neg_seq_len = item_crop_neg(seq.clone(), length.clone(), attribution_score,
                                                     self.config['neg_r'])
                # neg_seq, neg_seq_len = item_crop_neg(seq.clone(), length.clone(), attribution_score, 0.7)
            elif neg_switch[0] == 1:
                neg_seq, neg_seq_len = item_mask_neg(seq.clone(), length.clone(), attribution_score,
                                                     self.config['neg_r'])
                # neg_seq, neg_seq_len = item_mask_neg(seq.clone(), length.clone(), attribution_score, 0.5)
            elif neg_switch[0] == 2:
                neg_seq, neg_seq_len = item_reorder_neg(seq.clone(), length.clone(), attribution_score,
                                                        self.config['neg_r'])
                # neg_seq, neg_seq_len = item_reorder_neg(seq.clone(), length.clone(), attribution_score, 0.3)

            neg_seq_.append(neg_seq)
            neg_seq_len_.append(neg_seq_len)

        cur_data.update(Interaction({'aug1': torch.stack(aug_seq1), 'aug_len1': torch.stack(aug_len1),
                                     'aug2': torch.stack(aug_seq2), 'aug_len2': torch.stack(aug_len2),
                                     'aug_neg': torch.stack(neg_seq_), 'aug_neg_len': torch.stack(neg_seq_len_)}))

    def duorec_aug(self, cur_data, index):
        cur_same_target = self.same_target_index[index]
        null_index = []
        sample_pos = []
        for i, targets in enumerate(cur_same_target):
            # in case there is no same-target sequence
            # don't know why this happens since the filtering has been applied
            if len(targets) == 0:
                sample_pos.append(-1)
                null_index.append(i)
            else:
                sample_pos.append(np.random.choice(targets))
        sem_pos_seqs = self.static_item_id_list[sample_pos]
        sem_pos_lengths = self.static_item_length[sample_pos]
        if null_index:
            sem_pos_seqs[null_index] = cur_data['item_id_list'][null_index]
            sem_pos_lengths[null_index] = cur_data['item_length'][null_index]

        cur_data.update(Interaction({'sem_aug': sem_pos_seqs, 'sem_aug_lengths': sem_pos_lengths}))

    def duoxaiselrec_aug(self, cur_data, index):
        cur_same_target = self.same_target_index[index]

        union_target = self.union_target_index[index]
        intersect_target = self.intersect_target_index[index]

        null_index = []
        sample_pos = []
        for i, targets in enumerate(cur_same_target):
            if len(targets) == 0:
                sample_pos.append(-1)
                null_index.append(i)
            else:
                def rewight_distribution(original_distributon, temperature=0.5):
                    distribution = np.log(original_distributon) / temperature
                    distribution = np.exp(distribution)
                    return distribution / np.sum(distribution)

                cand_seqs = self.static_item_id_list[targets]
                cand_lengths = self.static_item_length[targets]
                cand_attributions = self.duorec_attribution_xai[targets]
                utility_scores_1 = []
                utility_scores_2 = []

                cand_sorted_indices = np.arange(len(cand_seqs))
                targets_c = targets

                for cand_i in cand_sorted_indices:
                    cand_length = cand_lengths[cand_i]
                    cand_attribution_x = cand_attributions[cand_i][:cand_length]

                    s_ids = intersect_target[i][cand_i]

                    if len(s_ids) == 0:
                        utility_x = 0.01
                    else:
                        utility_x = np.sum(cand_attribution_x[s_ids])

                    utility_scores_1.append(utility_x)

                    utility_sim = union_target[i][cand_i]
                    utility_scores_2.append(utility_sim)

                utility_scores = np.multiply(np.exp(np.array(utility_scores_1)), np.array(utility_scores_2))
                utility_p = softmax(utility_scores)

                utility_p = rewight_distribution(utility_p, self.config['duo_temp'])

                sample_pos.append(np.random.choice(targets_c, p=utility_p))
        sem_pos_seqs = self.static_item_id_list[sample_pos]
        sem_pos_lengths = self.static_item_length[sample_pos]
        if null_index:
            sem_pos_seqs[null_index] = cur_data['item_id_list'][null_index]
            sem_pos_lengths[null_index] = cur_data['item_length'][null_index]

        cur_data.update(Interaction({'sem_aug': sem_pos_seqs, 'sem_aug_lengths': sem_pos_lengths}))

    def update_xai_info(self, examples_attribution):
        self.examples_attribution_xai = examples_attribution
        self.dataset.inter_feat.update(Interaction({'attribution_scores': examples_attribution}))

        if self.config['method'] in ['DuoRec_XAUG', 'EC4SRec']:
            reverse_index = torch.sort(self.mapping_index)[-1]
            self.duorec_attribution_xai = examples_attribution[reverse_index]
            average_v = np.sum(self.duorec_attribution_xai) / torch.sum(self.static_item_length).item()
            tmp = np.where(self.duorec_attribution_xai > average_v, self.duorec_attribution_xai, 0.)
            self.duorec_attribution_xai_score = np.sum(tmp, axis=-1)


class NegSampleEvalDataLoader(NegSampleDataLoader):
    """:class:`NegSampleEvalDataLoader` is a dataloader for neg-sampling evaluation.
    It is similar to :class:`TrainDataLoader` which can generate negative items,
    and this dataloader also permits that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self._set_neg_sample_args(config, dataset, InputType.POINTWISE, config['eval_neg_sample_args'])
        if self.neg_sample_args['strategy'] == 'by':
            user_num = dataset.user_num
            dataset.sort(by=dataset.uid_field, ascending=True)
            self.uid_list = []
            start, end = dict(), dict()
            for i, uid in enumerate(dataset.inter_feat[dataset.uid_field].numpy()):
                if uid not in start:
                    self.uid_list.append(uid)
                    start[uid] = i
                end[uid] = i
            self.uid2index = np.array([None] * user_num)
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            for uid in self.uid_list:
                self.uid2index[uid] = slice(start[uid], end[uid] + 1)
                self.uid2items_num[uid] = end[uid] - start[uid] + 1
            self.uid_list = np.array(self.uid_list)

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config['eval_batch_size']
        if self.neg_sample_args['strategy'] == 'by':
            inters_num = sorted(self.uid2items_num * self.times, reverse=True)
            batch_num = 1
            new_batch_size = inters_num[0]
            for i in range(1, len(inters_num)):
                if new_batch_size + inters_num[i] > batch_size:
                    break
                batch_num = i + 1
                new_batch_size += inters_num[i]
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        self._set_neg_sample_args(config, self.dataset, InputType.POINTWISE, config['eval_neg_sample_args'])
        super().update_config(config)

    @property
    def pr_end(self):
        if self.neg_sample_args['strategy'] == 'by':
            return len(self.uid_list)
        else:
            return len(self.dataset)

    def _shuffle(self):
        self.logger.warnning('NegSampleEvalDataLoader can\'t shuffle')

    def _next_batch_data(self):
        if self.neg_sample_args['strategy'] == 'by':
            uid_list = self.uid_list[self.pr:self.pr + self.step]
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            for idx, uid in enumerate(uid_list):
                index = self.uid2index[uid]
                data_list.append(self._neg_sampling(self.dataset[index]))
                idx_list += [idx for i in range(self.uid2items_num[uid] * self.times)]
                positive_u += [idx for i in range(self.uid2items_num[uid])]
                positive_i = torch.cat((positive_i, self.dataset[index][self.iid_field]), 0)

            cur_data = cat_interactions(data_list)
            idx_list = torch.from_numpy(np.array(idx_list))
            positive_u = torch.from_numpy(np.array(positive_u))

            self.pr += self.step

            return cur_data, idx_list, positive_u, positive_i
        else:
            cur_data = self._neg_sampling(self.dataset[self.pr:self.pr + self.step])
            self.pr += self.step
            return cur_data, None, None, None


class FullSortEvalDataLoader(AbstractDataLoader):
    """:class:`FullSortEvalDataLoader` is a dataloader for full-sort evaluation. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.is_sequential = config['MODEL_TYPE'] == ModelType.SEQUENTIAL
        if not self.is_sequential:
            user_num = dataset.user_num
            self.uid_list = []
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            self.uid2positive_item = np.array([None] * user_num)
            self.uid2history_item = np.array([None] * user_num)

            dataset.sort(by=self.uid_field, ascending=True)
            last_uid = None
            positive_item = set()
            uid2used_item = sampler.used_ids
            for uid, iid in zip(dataset.inter_feat[self.uid_field].numpy(), dataset.inter_feat[self.iid_field].numpy()):
                if uid != last_uid:
                    self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
                    last_uid = uid
                    self.uid_list.append(uid)
                    positive_item = set()
                positive_item.add(iid)
            self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
            self.uid_list = torch.tensor(self.uid_list, dtype=torch.int64)
            self.user_df = dataset.join(Interaction({self.uid_field: self.uid_list}))

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _set_user_property(self, uid, used_item, positive_item):
        if uid is None:
            return
        history_item = used_item - positive_item
        self.uid2positive_item[uid] = torch.tensor(list(positive_item), dtype=torch.int64)
        self.uid2items_num[uid] = len(positive_item)
        self.uid2history_item[uid] = torch.tensor(list(history_item), dtype=torch.int64)

    def _init_batch_size_and_step(self):
        batch_size = self.config['eval_batch_size']
        if not self.is_sequential:
            batch_num = max(batch_size // self.dataset.item_num, 1)
            new_batch_size = batch_num * self.dataset.item_num
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    @property
    def pr_end(self):
        if not self.is_sequential:
            return len(self.uid_list)
        else:
            return len(self.dataset)

    def _shuffle(self):
        self.logger.warnning('FullSortEvalDataLoader can\'t shuffle')

    def _next_batch_data(self):
        if not self.is_sequential:
            user_df = self.user_df[self.pr:self.pr + self.step]
            uid_list = list(user_df[self.uid_field])

            history_item = self.uid2history_item[uid_list]
            positive_item = self.uid2positive_item[uid_list]

            history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
            history_i = torch.cat(list(history_item))

            positive_u = torch.cat([torch.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_item)])
            positive_i = torch.cat(list(positive_item))

            self.pr += self.step
            return user_df, (history_u, history_i), positive_u, positive_i
        else:
            interaction = self.dataset[self.pr:self.pr + self.step]
            inter_num = len(interaction)
            positive_u = torch.arange(inter_num)
            positive_i = interaction[self.iid_field]

            self.pr += self.step
            return interaction, None, positive_u, positive_i
