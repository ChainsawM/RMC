import os
import pickle
from glob import glob
import numpy as np
import argparse
import torch
import json
import time
import logging

# resolve importing path
from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))
sys.path.append(str(Path().absolute()))

from src.utils.tqdm_logging import Tqdm
from src.utils.helper import save_json, ensure_dir_exists, get_embedding_file_name, Dict2Class, load_json, \
    prepare_config
from src.utils.constants import scripts_folder, experiments_folder
from src.evaluating.metrics import calculate_metrics
from src.datautils.dataset import MCDataset
from src.modules.rankers import PLMRanker, BM25Ranker_whoosh
from src.modules.compute_papers_embedding import compute_paper_embeddings

logger = logging.getLogger(__name__)


def get_total_positive(_dict):
    return _dict['positive_ids']


def do_evaluate(_config,
                _dataset: MCDataset,
                _print=False,
                _ranker_type: str = 'plm',
                _save_predications=False, ):
    _paths = Dict2Class(_config.common_path)
    _model_para = Dict2Class(_config.model_para)

    _predications_results_save_path = _paths.result_folder + f"results_{_dataset.split}.json"

    embedding_path = _paths.embedding_folder + get_embedding_file_name()

    if _ranker_type == 'whoosh':
        ranker = BM25Ranker_whoosh(index_path=_paths.embedding_folder, corpus=_dataset.pid2info,
                                   _max_num_ref=_model_para.max_num_ref, bibrefs=_dataset.bibref2info)
    else:
        ranker = PLMRanker(embedding_path)

    k_list = _config.K_list
    positive_ids_list = []
    candidates_list = []
    qid2ranked = {}
    for query in Tqdm.tqdm(_dataset.samples, desc=f'Evaluating over {_dataset.split} samples'):
        qid = query['id']
        candidates, scores = ranker.get_top_n(qid=qid, n=_dataset.num_papers())

        if qid in candidates:
            temp_idx = candidates.index(qid)
            candidates.remove(qid)
            scores = np.delete(scores, temp_idx)

        positive_ids_list.append(get_total_positive(query))
        candidates_list.append(candidates)
        qid2ranked[qid] = candidates

    total_metrics = calculate_metrics(_candidates_list=candidates_list, _positive_ids_list=positive_ids_list,
                                      K_list=k_list)
    if _print:
        logger.info(total_metrics)
        print(total_metrics)

    _metric_save_path = _paths.result_folder + f"metrics_{_dataset.split}.json"
    ensure_dir_exists(_metric_save_path)
    save_json(total_metrics, _metric_save_path)
    logger.info('Wrote: {}'.format(_metric_save_path))

    if _save_predications:
        save_json(qid2ranked, _predications_results_save_path)

    return total_metrics


def evaluate_plm(_dataset_name, _experiment_name, _config_name, ):
    _config_path = scripts_folder + f'configs/{_dataset_name}/{_config_name}.config'
    _config = prepare_config(_config_path, _dataset_name, _experiment_name)

    saved_config_path = experiments_folder + f'{_dataset_name}/{_experiment_name}/log/config.json'
    saved_config = load_json(saved_config_path)
    _config.update(saved_config)

    _config = Dict2Class(_config)
    _train_para = Dict2Class(_config.train_para)
    _paths = Dict2Class(_config.common_path)

    print(f'Config:\n{_config}')
    compute_paper_embeddings(_config=_config, _model_path=_paths.best_model_path)

    test_dataset = MCDataset(name=_config.dataset, split='test', paper_batch_size=_config.paper_batch_size,
                             config=_config)
    test_result = do_evaluate(_config=_config, _dataset=test_dataset, _print=True, _save_predications=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment')
    parser.add_argument('--config_name', type=str, help='Name of the configuration')

    args = parser.parse_args()

    evaluate_plm(args.dataset_name, args.experiment_name, args.config_name)
