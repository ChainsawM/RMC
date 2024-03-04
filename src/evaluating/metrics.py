import numpy as np
from typing import List
from copy import deepcopy
# resolve importing path
from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))
sys.path.append(str(Path().absolute()))


def cal_map(__predication: List, __truth: List):
    hits = 0
    temp = 0
    for i in range(len(__predication)):
        if __predication[i] in __truth:
            hits += 1
            temp += hits / (i + 1)
    return temp / len(__truth)


def cal_mrr(__predication: List, __truth: List):
    for idx, _pred in enumerate(__predication):
        if _pred in __truth:
            return 1 / (idx + 1)
    return 0


def cal_dcg(__predication: List, __truth: List):
    dcg = 0
    for idx, _pred in enumerate(__predication):
        if _pred in __truth:
            dcg += 1 / np.log2(idx + 2)
    return dcg


def cal_ndcg(__predication: List, __truth: List):
    actual = cal_dcg(__predication, __truth)
    best = cal_dcg(__truth, __truth)
    return actual / best


def num_duplication(_list1, _list2):
    return len(set(_list1).intersection(_list2))


def calculate_metrics(_candidates_list, _positive_ids_list, K_list) -> dict:
    recall_at_K = {}
    precision_at_K = {}
    F_at_K = {}
    for K in K_list:
        recall_list = []
        precision_list = []

        for positive_ids, candidates in zip(_positive_ids_list, _candidates_list):
            hit_num = len(set(positive_ids) & set(candidates[:K]))
            recall_list.append(hit_num / len(set(positive_ids)))
            precision_list.append(hit_num / K)

        recall_at_K[K] = np.mean(recall_list)
        precision_at_K[K] = np.mean(precision_list)

        F_at_K[K] = 2 / (1 / (recall_at_K[K] + 1e-12) + 1 / (precision_at_K[K] + 1e-12))

    map_list = []
    mrr_list = []
    ndcg_list = []
    for positive_ids, candidates in zip(_positive_ids_list, _candidates_list):
        map_list.append(cal_map(candidates, positive_ids))
        mrr_list.append(cal_mrr(candidates, positive_ids))
        ndcg_list.append(cal_ndcg(candidates, positive_ids))
    map = np.mean(map_list)
    mrr = np.mean(mrr_list)
    ndcg = np.mean(ndcg_list)

    _metrics = {
        'map': map,
        'mrr': mrr,
        'ndcg': ndcg,
        'recall': recall_at_K,
        'precision': precision_at_K,
        'f1': F_at_K,
    }
    return _metrics
