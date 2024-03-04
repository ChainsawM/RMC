import pickle
import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from transformers import AutoModel, AutoTokenizer

# resolve importing path
from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))
sys.path.append(str(Path().absolute()))

from src.utils.helper import prepare_config, Dict2Class, get_embedding_file_name, get_direct_lm_name_and_path, \
    ensure_dir_exists
from src.utils.tqdm_logging import Tqdm
from src.datautils.dataset import MCDataset
from src.modules.models import PLMEncoder

logger = logging.getLogger(__name__)


def compute_paper_embeddings(_config,
                             dataset: MCDataset = None,
                             _model=None,
                             _encoder_gpu_list=[0],
                             _model_path=None,
                             _print=False, _nn_para=None):
    _paths = Dict2Class(_config.common_path)
    _model_para = Dict2Class(_config.model_para)

    device = torch.device("cuda:%d" % (_encoder_gpu_list[0])
                          if len(_encoder_gpu_list) > 0 and torch.cuda.is_available() else "cpu")

    embedding_save_path = _paths.embedding_folder + get_embedding_file_name()

    if _model is not None:
        encoder = _model
    else:
        if _model_path is not None:
            ckpt_name = _model_path
        else:
            try:
                if not os.path.exists(_paths.model_folder):
                    logger.warning(f'{_paths.model_folder} does not exist!')
                    ckpt_name = None
                else:
                    # model_batch_*.pt for the newest model
                    ckpt_list = glob(_paths.model_folder + "*.pt")
                    if len(ckpt_list) > 0:
                        ckpt_list.sort(key=os.path.getmtime)
                        ckpt_name = ckpt_list[-1]
                    else:
                        logger.warning(f'{_paths.model_folder} is empty!')
                        ckpt_name = None
            except:
                ckpt_name = None

        if ckpt_name is not None:
            if _print:
                logger.info(f'Loading {ckpt_name} to compute papers embeddings.')

            plm_name, plm_path = get_direct_lm_name_and_path(_config)
            encoder = PLMEncoder(plm_path=plm_path, embed_dim=_model_para.embed_dim,
                                 max_length=_model_para.max_seq_len,
                                 device=device,
                                 way_ref=_model_para.way_ref,
                                 with_linear=_model_para.with_linear,
                                 ratio_ref=_model_para.ratio_ref,)
            ckpt = torch.load(ckpt_name, map_location="cpu")
            encoder.load_state_dict(ckpt["document_encoder"])

        else:
            logger.error(f'No model is assigned and no model is available in {_paths.model_folder}!')
            raise FileNotFoundError(f'No model is assigned and no model is available in {_paths.model_folder}!')

    encoder.eval()

    if dataset is None:
        dataset = MCDataset(name=_config.dataset, split='test', paper_batch_size=_config.paper_batch_size,
                            config=_config)

    index_to_id_mapper = {}
    id_to_index_mapper = {}
    total_embeddings = []
    count = 0
    for batch_contents, batch_refs, batch_ids in tqdm(dataset.paper_batches(), desc='Compute paper embeddings',
                                                      total=dataset.num_papers() // dataset.paper_batch_size):
        batch = {
            'batch_contents': batch_contents,
            'batch_refs': batch_refs,
        }
        with torch.no_grad():
            batch_embeddings = encoder(batch)
        for paper_id, embedding in zip(batch_ids, batch_embeddings.unbind()):
            index_to_id_mapper[count] = paper_id
            id_to_index_mapper[paper_id] = count
            count += 1
            total_embeddings.append(embedding.detach().cpu().numpy().tolist())
    total_embeddings = np.array(total_embeddings)

    with open(ensure_dir_exists(embedding_save_path), "wb") as f:
        pickle.dump({
            "index_to_id_mapper": index_to_id_mapper,
            "id_to_index_mapper": id_to_index_mapper,
            "embedding": total_embeddings
        }, f, -1)
    if _print:
        logger.info(f'Wrote: {embedding_save_path}')
    encoder.train()


if __name__ == "__main__":
    pass
