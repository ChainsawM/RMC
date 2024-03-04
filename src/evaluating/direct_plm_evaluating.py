import logging
import torch
import argparse
from transformers import AutoTokenizer
# resolve importing path
from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))
sys.path.append(str(Path().absolute()))

from src.utils.constants import scripts_folder, model2online_path
from src.utils.helper import prepare_config, Dict2Class, create_folders, get_direct_lm_name_and_path, prepare_logging
from src.evaluating.evaluate import do_evaluate
from src.datautils.dataset import MCDataset
from src.modules.compute_papers_embedding import compute_paper_embeddings
from src.modules.models import PLMEncoder

logger = logging.getLogger(__name__)


def get_ratio_str(_ratio: float):
    if _ratio < 1:
        return str(int(_ratio * 10))
    else:
        return 'A'


def evaluate_direct_plm(_model_name: str, _dataset_name: str, way_ref: str, with_linear=False, ratio_ref=None):
    config_name = 'example'
    experiment_name = f'{_model_name}_{way_ref}_static'

    if ratio_ref is not None:
        experiment_name = experiment_name.replace('_static', f'_ratio{get_ratio_str(ratio_ref)}_static')

    _config_path = scripts_folder + f'configs/{_dataset_name}/{config_name}.config'

    _config = prepare_config(_config_path, _dataset_name, experiment_name)
    _config['model'] = _model_name
    _config['dataset'] = _dataset_name
    _config['model_para']['way_ref'] = way_ref
    logger.info(f'Config:\n{_config}')
    _config = Dict2Class(_config)

    _train_para = Dict2Class(_config.train_para)
    _paths = Dict2Class(_config.common_path)
    _model_para = Dict2Class(_config.model_para)
    create_folders(_config.common_path)
    prepare_logging(_paths.log_folder)

    # prepare the model
    plm_name, plm_path = get_direct_lm_name_and_path(_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if plm_name and plm_path:
        model = PLMEncoder(plm_path=plm_path, embed_dim=_model_para.embed_dim,
                           max_length=_model_para.max_seq_len,
                           device=device, way_ref=_model_para.way_ref,
                           with_linear=_model_para.with_linear,
                           ratio_ref=ratio_ref)
        model.eval()
        plm_tokenizer = AutoTokenizer.from_pretrained(plm_path)
    else:
        raise ValueError('No specific pretrained language model is selected in config!')

    # prepare the dataset
    dataset = MCDataset(name=_config.dataset, split='test', paper_batch_size=_config.paper_batch_size,
                        config=_config)

    compute_paper_embeddings(_config=_config, dataset=dataset, _model=model)

    do_evaluate(_config=_config, _dataset=dataset, _save_predications=True, _print=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    parser.add_argument("--way_ref", type=str, help="the way of dealing with the reference")
    parser.add_argument("--with_linear", action="store_true", help="whether to use linear layer or not")
    parser.add_argument("--ratio_ref", type=float, default=None, help="weight of the reference")

    args = parser.parse_args()

    evaluate_direct_plm(_model_name=args.model_name,
                        _dataset_name=args.dataset_name,
                        way_ref=args.way_ref,
                        with_linear=args.with_linear,
                        ratio_ref=args.ratio_ref)
