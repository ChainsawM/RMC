import json
import os
import logging
from glob import glob
import argparse

# resolve importing path
from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))
sys.path.append(str(Path().absolute()))

from src.utils.constants import scripts_folder
from src.utils.helper import prepare_config, Dict2Class, create_folders, prepare_logging, load_json, save_json, \
    get_best_model, ensure_dir_exists
from src.modules.compute_papers_embedding import compute_paper_embeddings
from src.modules.get_prefetched_ids import get_prefetched_ids
from src.training.trainer import do_training
from src.evaluating.evaluate import do_evaluate
from src.datautils.dataset import MCDataset

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_one_experiment(_model_name, _dataset_name, _config_name, _way_ref, _experiment_name, _ratio_ref=None):
    _config_path = scripts_folder + f'configs/{_dataset_name}/{_config_name}.config'

    _config = prepare_config(_config_path, _dataset_name, _experiment_name)
    _config['model'] = _model_name
    _config['dataset'] = _dataset_name
    _config['model_para']['way_ref'] = _way_ref
    if 'ratio_ref' not in _config['model_para']:
        _config['model_para']['ratio_ref'] = _ratio_ref
    logger.info(f'Config:\n{_config}')
    save_json(_config, ensure_dir_exists(_config['common_path']['log_folder'] + "config.json"))

    _config = Dict2Class(_config)

    _train_para = Dict2Class(_config.train_para)
    _paths = Dict2Class(_config.common_path)

    create_folders(_config.common_path)
    prepare_logging(_paths.log_folder)

    # TODO gpu setting
    gpu_list = list(range(0, _config.n_device))

    if not os.path.exists(_paths.prefetched_train_path):
        os.system("cp " + _paths.initial_train_path + " " + _paths.prefetched_train_path)
    if not os.path.exists(_paths.prefetched_val_path):
        os.system("cp " + _paths.initial_val_path + " " + _paths.prefetched_val_path)

    val_metrics = []

    # restore from checkpoint
    if os.path.exists(_paths.model_folder) and len(glob(_paths.model_folder + "/*.pt")) > 0:
        if _train_para.restore_old_checkpoint:
            logger.info("Found existing model, compute the embedding and get prefetched ids!")
            compute_paper_embeddings(_config=_config)
            get_prefetched_ids(_config=_config, _split='train')
            get_prefetched_ids(_config=_config, _split='val')
            if os.path.exists(_paths.result_folder + 'val_metrics.json'):
                val_metrics = load_json(_paths.result_folder + 'val_metrics.json')
        else:
            raise EnvironmentError(f"Can't start a new experiment while {_paths.model_folder} is not empty")

    # do training loop
    for loop_idx in range(_train_para.num_training_loop):
        trained_plm = do_training(_config=_config, _loop_idx=loop_idx, _val_metrics=val_metrics)
        logger.info('-' * 20 + f"Finish training loop {loop_idx}/{_train_para.num_training_loop - 1}")
        if loop_idx < _train_para.num_training_loop - 1:
            if _train_para.max_num_iterations % _train_para.validate_every != 0:
                compute_paper_embeddings(_config=_config, _model_path=_paths.best_model_path)
            get_prefetched_ids(_config=_config, _split='train')
            get_prefetched_ids(_config=_config, _split='val')

    # get best model from savings according to val
    best_ckpt = get_best_model(val_metrics)
    val_metrics.insert(0, {'best_ckpt': best_ckpt})
    json.dump(val_metrics, open(_paths.result_folder + 'val_metrics.json', "w"), ensure_ascii=False, indent=1)

    # do testing
    logger.info('-' * 20 + f'Finish {_train_para.num_training_loop} training loops, testing best model {best_ckpt}!')
    compute_paper_embeddings(_config=_config, _model_path=_paths.best_model_path)

    test_dataset = MCDataset(name=_config.dataset, split='test', paper_batch_size=_config.paper_batch_size,
                             config=_config)
    test_result = do_evaluate(_config=_config, _dataset=test_dataset, _print=True,
                              _save_predications=False)

    logger.info('Experiment finish!!!')


def duplicate_run_experiment(_model_name, _dataset_name, _config_name, _way_ref, _experiment_name, _num_run,
                             _ratio_ref=None):
    for idx in range(_num_run):
        run_experiment_suffix = f'{_experiment_name}_{idx}'
        run_one_experiment(_model_name=_model_name, _dataset_name=_dataset_name, _config_name=_config_name,
                           _experiment_name=run_experiment_suffix, _way_ref=_way_ref, _ratio_ref=_ratio_ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='scincl')
    parser.add_argument("-d", "--dataset", type=str, default='base')
    parser.add_argument("-c", "--config_name", type=str)
    parser.add_argument("-s", "--experiment_suffix", type=str)
    parser.add_argument("-w", "--way_ref", type=str, default='avg')
    parser.add_argument("-n", "--num_run", type=int, default=3)
    parser.add_argument("-r", "--ratio_ref", type=float, default=0.6)
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset
    config_name = args.config_name
    experiment_name = args.experiment_suffix
    way_ref = args.way_ref
    num_run = args.num_run
    ratio_ref = args.ratio_ref

    if num_run is None:
        run_one_experiment(_model_name=model_name, _dataset_name=dataset_name, _config_name=config_name,
                           _experiment_name=experiment_name, _way_ref=way_ref, _ratio_ref=ratio_ref)
    else:
        duplicate_run_experiment(_model_name=model_name, _dataset_name=dataset_name, _config_name=config_name,
                                 _experiment_name=experiment_name, _way_ref=way_ref, _num_run=num_run,
                                 _ratio_ref=ratio_ref)
