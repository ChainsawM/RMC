import logging
from tqdm import tqdm
from glob import glob
import json
import os
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from functools import lru_cache
from nltk.corpus import stopwords
from typing import List, Dict, Union

# resolve importing path
from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))
sys.path.append(str(Path().absolute()))

from src.utils.constants import data_folder, experiments_folder, model2online_path, model_hub_folder, default_K_list

logger = logging.getLogger(__name__)


def load_json(_path: str) -> Union[dict]:
    with open(_path, 'r', encoding='utf-8') as _fp:
        return json.load(_fp)


def save_json(_dict, _path: str):
    with open(_path, 'w', encoding='utf-8') as _fp:
        json.dump(_dict, _fp, ensure_ascii=False, indent=1)


def load_jsonl(_path: str) -> list:
    with open(_path, 'r', encoding='utf-8') as _fp:
        return [json.loads(line) for line in tqdm(_fp)]


def save_jsonl(_list, _path: str):
    with open(_path, 'w', encoding='utf-8') as _fp:
        for doc in _list:
            _fp.write(json.dumps(doc) + '\n')


def ensure_dir_exists(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return path


class SentenceTokenizer:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lemmatizer = WordNetLemmatizer()
        self.general_stopwords = set(stopwords.words('english')) | {"<num>", "<cit>"}

    @lru_cache(100000)
    def lemmatize(self, w):
        return self.lemmatizer.lemmatize(w)

    def tokenize(self, sen, remove_stopwords=True):
        if remove_stopwords:
            sen = " ".join([w for w in sen.lower().split() if w not in self.general_stopwords])
        else:
            sen = sen.lower()
        sen = " ".join([self.lemmatize(w) for w in self.tokenizer.tokenize(sen)])
        return sen


class JaccardSim:
    def __init__(self):
        self.sent_tok = SentenceTokenizer()

    def compute_sim(self, textA, textB):
        textA_words = set(self.sent_tok.tokenize(textA.lower(), remove_stopwords=True).split())
        textB_words = set(self.sent_tok.tokenize(textB.lower(), remove_stopwords=True).split())

        AB_words = textA_words.intersection(textB_words)
        return float(len(AB_words) / (len(textA_words) + len(textB_words) - len(AB_words) + 1e-12))


def prepare_config(_path, dataset_name, _experiment_name) -> Dict:
    _run_config = json.load(open(_path, encoding='utf-8'))

    # prepare data path
    _dataset = dataset_name
    _original_dataset_folder = f'{data_folder}{_dataset}/'

    _papers_path = f'{_original_dataset_folder}papers.json'
    _bibref_path = f'{_original_dataset_folder}bibref2info.json'

    _initial_train_path = f'{data_folder}train.json'
    _initial_val_path = f'{data_folder}val.json'
    _initial_test_path = f'{data_folder}test.json'

    # prepare experiment path
    _run_config['experiment_name'] = _experiment_name
    _experiment_folder = f'{experiments_folder}{_dataset}/{_experiment_name}/'
    _experiment_data_folder = f'{_experiment_folder}dataset/'
    _model_folder = f'{_experiment_folder}model/'
    _board_folder = f'{_experiment_folder}board/'
    _embedding_folder = f'{_experiment_folder}embedding/'
    _result_folder = f'{_experiment_folder}results/'
    _log_folder = f'{_experiment_folder}log/'

    _prefetched_train_path = f'{_experiment_data_folder}train_with_prefetched_ids.json'
    _prefetched_val_path = f'{_experiment_data_folder}val_with_prefetched_ids.json'
    _prefetched_test_path = f'{_experiment_data_folder}test_with_prefetched_ids.json'

    common_path = {
        'original_dataset_folder': _original_dataset_folder,
        'papers_path': _papers_path,
        'bibref_path': _bibref_path,

        'prefetched_train_path': _prefetched_train_path,
        'prefetched_val_path': _prefetched_val_path,
        'prefetched_test_path': _prefetched_test_path,

        'initial_train_path': _initial_train_path,
        'initial_val_path': _initial_val_path,
        'initial_test_path': _initial_test_path,

        'experiment_folder': _experiment_folder,
        'board_folder': _board_folder,
        'model_folder': _model_folder,
        'experiment_data_folder': _experiment_data_folder,
        'embedding_folder': _embedding_folder,
        'result_folder': _result_folder,
        'log_folder': _log_folder,

        'best_model_path': _model_folder + 'model_best.pt',

    }
    _run_config['common_path'] = common_path
    return _run_config


class Dict2Class(object):
    """
        common_paths:
            'original_dataset_folder',
            'papers_path',
            'bibref_path',

            'prefetched_train_path',
            'prefetched_val_path',
            'prefetched_test_path',

            'initial_train_path',
            'initial_val_path',
            'initial_test_path',

            'experiment_folder',
            'board_folder',
            'model_folder',
            'experiment_data_folder',
            'embedding_folder',
            'result_folder',
            'log_folder',

            'best_model_path'
    """

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def create_folders(_lst: Dict):
    for _folder in _lst.values():
        if _folder.endswith('/'):
            if not os.path.exists(_folder):
                os.makedirs(_folder)
                logger.info(f'Folder created: {_folder}')
            else:
                logger.info(f'Folder already exists: {_folder}')
        else:
            pass


def prepare_logging(_log_folder):
    """
    based on allennlp prepare_logging
    """
    root_logger = logging.getLogger()
    _level = logging.INFO
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_handler = logging.FileHandler(_log_folder + f'running_{current_time}.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)
    for handler in [file_handler, stdout_handler, stderr_handler]:
        handler.setFormatter(formatter)
    root_logger.handlers.clear()

    file_handler.setLevel(_level)
    stdout_handler.setLevel(_level)
    stderr_handler.setLevel(logging.ERROR)
    root_logger.setLevel(_level)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)


def get_direct_lm_name_and_path(_config, online=False):
    _name = _config.model

    if 'online' in model_hub_folder:
        online = True

    if online:
        if _name in model2online_path:
            _path = model2online_path[_name]
        else:
            raise ValueError(f'{_name} is not supported for online!')
    else:
        _path = model_hub_folder + f'{_name}'

    return _name, _path


def worker_init_fn(x):
    return [np.random.seed(int(time.time()) + x), torch.manual_seed(int(time.time()) + x)]


def load_model(model_folder, return_ckpt_name=False):
    ckpt_name = None
    ckpt_list = glob(model_folder + "/*.pt")
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getmtime)
        ckpt_name = ckpt_list[-1]
        ckpt = torch.load(ckpt_name, map_location=torch.device('cpu'))
    else:
        ckpt = None
    if return_ckpt_name:
        return ckpt, str(ckpt_name)
    else:
        return ckpt


def save_model(module_dicts, save_name, max_to_keep=0, overwrite=True):
    folder_path = os.path.dirname(os.path.abspath(save_name))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    state_dicts = {}
    for key in module_dicts.keys():
        if isinstance(module_dicts[key], nn.DataParallel):
            state_dicts[key] = module_dicts[key].module.state_dict()
        elif isinstance(module_dicts[key], nn.Module):
            state_dicts[key] = module_dicts[key].state_dict()
        else:
            state_dicts[key] = module_dicts[key]

    if os.path.exists(save_name):
        if overwrite:
            os.remove(save_name)
            torch.save(state_dicts, save_name)
        else:
            print("Warning: checkpoint file already exists!")
            return
    else:
        torch.save(state_dicts, save_name)

    if max_to_keep > 0:
        pt_file_list = glob(folder_path + "/*.pt")
        pt_file_list.sort(key=lambda x: os.path.getmtime(x))
        for idx in range(len(pt_file_list) - max_to_keep):
            if 'model_best' not in pt_file_list[idx]:
                os.remove(pt_file_list[idx])


def get_embedding_file_name() -> str:
    return "paper_embedding.pkl"


def get_best_model(_metrics_dict: List[Dict], key='ndcg', k=10):
    value_lst = []
    names = []
    for info in _metrics_dict:
        if 'ckpt_name' not in info:
            continue
        name = info['ckpt_name']
        _metrics = info['metrics']
        names.append(name)
        if not isinstance(_metrics[key], list):
            value_lst.append(_metrics[key])
        else:
            if k in _metrics[key]:
                value_lst.append(_metrics[key][k])
            else:
                value_lst.append(_metrics[key][str(k)])

    return names[value_lst.index(max(value_lst))]
