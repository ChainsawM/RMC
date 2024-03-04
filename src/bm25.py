import pickle
import numpy as np
import logging
from tqdm import tqdm
import os

from whoosh.index import create_in
from whoosh.fields import *
# resolve importing path
from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))
sys.path.append(str(Path().absolute()))
from src.utils.helper import ensure_dir_exists, SentenceTokenizer, get_embedding_file_name, prepare_config, Dict2Class, \
    create_folders
from src.utils.constants import experiments_folder, scripts_folder
from src.evaluating.evaluate import do_evaluate
from src.datautils.dataset import MCDataset
from src.modules.rankers import schema, get_bib_titles

logger = logging.getLogger(__name__)


# !!!
# Some errors are raised when using Whoosh 2.7.4, we edit the code according to:
# https://github.com/whoosh-community/whoosh/pull/521/commits/597a28484074b299abe9f48954f704a299e365c0

def do_get_inverted_index_whoosh(_dataset: MCDataset, _embedding_save_folder: str, _max_num_ref=0):
    bm25_index = create_in(ensure_dir_exists(_embedding_save_folder), schema)
    writer = bm25_index.writer()

    for index, (pid, paper) in enumerate(tqdm(_dataset.pid2info.items(), desc='Whoosh indexing')):

        bib_titles = get_bib_titles(_max_num_ref=_max_num_ref, paper=paper, bibrefs=_dataset.bibref2info)

        writer.add_document(
            id=pid,
            title=paper["title"],
            abstract=paper["abstract"],
            bib_titles=bib_titles
        )
    writer.commit()


def evaluate_bm25(_dataset_name, _max_num_ref=0):
    _model_name = 'whoosh'
    _ranker = 'whoosh'

    experiment_name = f'{_model_name}'

    _config_path = scripts_folder + f'configs/{_dataset_name}/bm25.config'
    run_config = prepare_config(_config_path, _dataset_name, experiment_name)
    run_config['model'] = _model_name
    run_config['dataset'] = _dataset_name
    run_config = Dict2Class(run_config)
    _paths = Dict2Class(run_config.common_path)

    create_folders(run_config.common_path)

    os.system("cp " + _config_path + " " + _paths.log_folder + "config.json")

    dataset = MCDataset(run_config, _dataset_name, 'test')

    do_get_inverted_index_whoosh(_dataset=dataset, _embedding_save_folder=_paths.embedding_folder, _max_num_ref=_max_num_ref)

    do_evaluate(_config=run_config, _dataset=dataset, _ranker_type=_ranker, _save_predications=True, _print=True)


if __name__ == "__main__":
    for ds in ['base', 'extended']:
        evaluate_bm25(ds, _max_num_ref=50)
