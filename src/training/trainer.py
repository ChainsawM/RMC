import pickle
import copy
import argparse
import logging
import torch
import os
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from torch.optim import AdamW

# resolve importing path
from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))
sys.path.append(str(Path().absolute()))

from src.utils.helper import prepare_config, Dict2Class, load_json, get_direct_lm_name_and_path, worker_init_fn, \
    load_model, save_model, get_best_model
from src.modules.models import PLMEncoder
from src.modules.losses import TripletLoss
from src.datautils.dataset import MCDataset, MCLoader
from src.evaluating.evaluate import do_evaluate
from src.modules.compute_papers_embedding import compute_paper_embeddings
from src.utils.tqdm_logging import Tqdm

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_iteration(_train_para, batch, device, document_encoder, triplet_loss: TripletLoss, optimizer):
    class_label_list = torch.tensor(batch["class_label_list"]).to(device)
    irrelevance_level_list = torch.tensor(batch["irrelevance_level_list"]).to(device)

    doc_embedding = document_encoder(batch)

    loss = triplet_loss(doc_embedding, class_label_list, irrelevance_level_list,
                        _train_para.positive_irrelevance_levels,
                        _train_para.similarity)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def validate_iteration(_train_para, batch, device, document_encoder, triplet_loss):
    class_label_list = torch.tensor(batch["class_label_list"]).to(device)
    irrelevance_level_list = torch.tensor(batch["irrelevance_level_list"]).to(device)

    with torch.no_grad():
        doc_embedding = document_encoder(batch)

        loss = triplet_loss(doc_embedding, class_label_list, irrelevance_level_list,
                            _train_para.positive_irrelevance_levels,
                            _train_para.similarity)
    return loss.item()


def is_train_para(_name: str, layers_kept=0):
    unfreeze_layers = [f'layer.{idx}' for idx in range(12)] + ['bert.pooler', 'out.']

    if not layers_kept:
        return 'bert' not in _name
    elif layers_kept <= 13:
        unfreeze_layers = unfreeze_layers[-layers_kept - 1:]
        if 'bert' not in _name:
            return True
        for one in unfreeze_layers:
            if one in _name:
                return True
        return False
    else:
        raise ValueError(f'Bert has only 13 layers!')


def do_training(_config, _loop_idx: int, _val_metrics: list):
    # prepare parameters
    _paths = Dict2Class(_config.common_path)
    _train_para = Dict2Class(_config.train_para)
    _model_para = Dict2Class(_config.model_para)

    gpu_list = list(range(0, _config.n_device))
    if _config.gpu_list is not None:
        assert len(_config.gpu_list) == _config.n_device
    else:
        _config.gpu_list = np.arange(_config.n_device).tolist()
    device = torch.device("cuda:%d" % (_config.gpu_list[0]) if (torch.cuda.is_available() and len(_config.gpu_list) > 0)
                          else "cpu")

    logger.info(f'Now running training loop {_loop_idx}/{_train_para.num_training_loop - 1}')

    # corpus and board_writer
    writer_train = SummaryWriter(f'{_paths.board_folder}train_loop{_loop_idx}')

    # build model frame
    plm_name, plm_path = get_direct_lm_name_and_path(_config)
    if plm_name and plm_path:
        document_encoder = PLMEncoder(plm_path=plm_path, embed_dim=_model_para.embed_dim,
                                      max_length=_model_para.max_seq_len,
                                      device=device, way_ref=_model_para.way_ref,
                                      with_linear=_model_para.with_linear,
                                      ratio_ref=_model_para.ratio_ref,)
        plm_tokenizer = AutoTokenizer.from_pretrained(plm_path)
    else:
        raise ValueError('No specific pretrained language model is selected in config!')

    # data and dataloader
    train_dataset = MCDataset(name=_config.dataset,
                              config=_config,
                              split='train',
                              tokenizer=plm_tokenizer,

                              max_seq_len=_model_para.max_seq_len,
                              max_num_ref=_model_para.max_num_ref,

                              max_n_sample=_train_para.max_n_sample,
                              max_n_positive=_train_para.max_n_positive,
                              max_n_hard_negative=_train_para.max_n_hard_negative,
                              max_n_easy_negative=_train_para.max_n_easy_negative,

                              paper_batch_size=_config.paper_batch_size
                              )
    train_dataloader = MCLoader(train_dataset,
                                batch_size=_train_para.n_query_per_batch,
                                shuffle=True,
                                worker_init_fn=worker_init_fn,
                                num_workers=_config.num_workers,
                                drop_last=True,
                                pin_memory=True)

    val_dataset = MCDataset(name=_config.dataset,
                            config=_config,
                            split='val',
                            tokenizer=plm_tokenizer,

                            max_seq_len=_model_para.max_seq_len,
                            max_num_ref=_model_para.max_num_ref,

                            max_n_sample=_train_para.max_n_sample,
                            max_n_positive=_train_para.max_n_positive,
                            max_n_hard_negative=_train_para.max_n_hard_negative,
                            max_n_easy_negative=_train_para.max_n_easy_negative,

                            paper_batch_size=_config.paper_batch_size
                            )
    val_dataloader = MCLoader(val_dataset,
                              batch_size=_train_para.n_query_per_batch,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=_config.num_workers,
                              drop_last=True,
                              pin_memory=True)

    # restore most recent checkpoint
    if _train_para.restore_old_checkpoint:
        ckpt, ckpt_path = load_model(_paths.model_folder, return_ckpt_name=True)
    else:
        ckpt = None
    # load before stored model state
    if ckpt is not None:
        document_encoder.load_state_dict(ckpt["document_encoder"])
        logger.info(f"Model restored from {ckpt_path}")

    # move to device
    document_encoder.to(device)

    # multi-gpus
    if device.type == "cuda" and _config.n_device > 1:
        document_encoder = nn.DataParallel(document_encoder, _config.gpu_list)
        model_parameters = [(name, par) for name, par in document_encoder.module.named_parameters() if
                            is_train_para(name, _model_para.num_kept_layers) and par.requires_grad]
    else:
        model_parameters = [(name, par) for name, par in document_encoder.named_parameters() if
                            is_train_para(name, _model_para.num_kept_layers) and par.requires_grad]
    print(f"Parameters to train: ", '\n'.join([one[0] for one in model_parameters]))

    # optimizer
    optimizer = AdamW([one[1] for one in model_parameters], lr=_train_para.initial_learning_rate,
                      weight_decay=_train_para.l2_weight)

    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("optimizer restored!")

    current_batch = 0
    if ckpt is not None:
        current_batch = ckpt["current_batch"]
        logger.info("current_batch restored!")

    running_losses = []
    triplet_loss = TripletLoss(_train_para.base_margin)

    for count in Tqdm.tqdm(range(_train_para.max_num_iterations)):
        current_batch += 1
        already_saved = False
        document_encoder.train()

        batch = train_dataloader.get_next()

        loss = train_iteration(_train_para, batch, device, document_encoder, triplet_loss, optimizer)

        running_losses.append(loss)
        writer_train.add_scalar('train_loss', loss, _loop_idx * _train_para.max_num_iterations + count)
        writer_train.add_scalar('train_loss_mean', np.mean(running_losses),
                                _loop_idx * _train_para.max_num_iterations + count)

        if current_batch % _train_para.print_every == 0:
            logger.info("[batch: %05d] Training loss: %.4f" % (current_batch, np.mean(running_losses)))
            os.system("nvidia-smi > %s/gpu_usage.log" % (_paths.log_folder))
            running_losses = []

        if _train_para.validate_every is not None and current_batch % _train_para.validate_every == 0:
            running_losses_val = []
            for _ in range(_train_para.num_validation_iterations):
                batch = val_dataloader.get_next()
                loss = validate_iteration(_train_para, batch, device, document_encoder, triplet_loss)
                running_losses_val.append(loss)
            logger.info("[batch: %05d] Validation loss: %.4f" % (current_batch, np.mean(running_losses_val)))

            save_model({
                "current_batch": current_batch,
                "document_encoder": document_encoder,
                "optimizer": optimizer.state_dict()
            }, _paths.model_folder + "/model_batch_%d.pt" % (current_batch), _train_para.max_num_checkpoints)
            already_saved = True

            writer_train.add_scalar('val_loss_mean', np.mean(running_losses_val),
                                    _loop_idx * _train_para.max_num_iterations + count)

            compute_paper_embeddings(_config=_config, dataset=train_dataset, _model=document_encoder, _print=False)

            temp_val_metric = do_evaluate(_config=_config, _dataset=val_dataset, _print=False,
                                          _save_predications=False)

            temp_val_metric['val_loss'] = np.mean(running_losses_val)
            val_metric = {
                'metrics': temp_val_metric,
                'ckpt_name': _paths.model_folder + "/model_batch_%d.pt" % (current_batch)
            }
            logger.info(val_metric)
            _val_metrics.append(val_metric)

            best_ckpt = get_best_model(_val_metrics)
            if 'model_best.pt' not in os.listdir(_paths.model_folder) or f'model_batch_{current_batch}.pt' in best_ckpt:
                save_model({
                    "current_batch": current_batch,
                    "document_encoder": document_encoder,
                    "optimizer": optimizer.state_dict()
                }, _paths.best_model_path, _train_para.max_num_checkpoints)
                logger.info(f"Wrote: model_best.pt at batch-{current_batch}")

        if not already_saved and current_batch % _train_para.save_every == 0:
            save_model({
                "current_batch": current_batch,
                "document_encoder": document_encoder,
                "optimizer": optimizer.state_dict()
            }, _paths.model_folder + "/model_batch_%d.pt" % (current_batch), _train_para.max_num_checkpoints)
            already_saved = True
            logger.info(f"Wrote: model_batch_{current_batch}.pt")

    return document_encoder


if __name__ == "__main__":
    pass
