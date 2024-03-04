import os
from glob import glob
import numpy as np
import json
import logging
import torch

# resolve importing path
from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))
sys.path.append(str(Path().absolute()))

from src.utils.helper import save_json, get_embedding_file_name, Dict2Class
from src.datautils.dataset import MCDataset
from src.modules.rankers import PLMRanker

logger = logging.getLogger(__name__)


def get_prefetched_ids(_config,
                       _split: str):
    _paths = Dict2Class(_config.common_path)
    _model_para = Dict2Class(_config.model_para)

    _dataset = MCDataset(name=_config.dataset, split=_split, paper_batch_size=_config.paper_batch_size, config=_config)

    if 'val' == _split:
        output_corpus_path = _paths.prefetched_val_path
    else:
        output_corpus_path = _paths.prefetched_train_path

    embedding_path = _paths.embedding_folder + get_embedding_file_name()
    ranker = PLMRanker(embedding_path)
    top_K = _config.num_knn

    ans = []
    for example in _dataset.samples:

        citing_id = example["id"]
        candidates, scores = ranker.get_top_n(qid=citing_id, n=top_K)

        if citing_id in set(candidates):
            candidates.remove(citing_id)
        example["prefetched_ids"] = candidates
        ans.append(example)

    save_json(ans, output_corpus_path)

    logger.info(f'Wrote: {output_corpus_path}')


if __name__ == "__main__":
    pass
