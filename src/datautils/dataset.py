import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# resolve importing path
from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))
sys.path.append(str(Path().absolute()))

from src.utils.helper import load_jsonl, load_json, save_json, JaccardSim, worker_init_fn, Dict2Class
from src.utils.constants import data_folder, base_folder, extended_folder

logger = logging.getLogger(__name__)


class MCDataset(Dataset):
    def __init__(self,
                 config,
                 name: str,
                 split: str,
                 tokenizer=None,

                 max_seq_len=512,
                 max_num_ref=50,

                 max_n_sample=12,
                 max_n_positive=1,
                 max_n_hard_negative=1,
                 max_n_easy_negative=2,

                 paper_batch_size=8,
                 ):

        if 'base' in name:
            dataset_folder = base_folder
        elif 'extended' in name:
            dataset_folder = extended_folder
        else:
            raise NotImplementedError(f'There is no dataset named {name} (only base, extended, and debug).')

        self.name = name
        self.split = split
        self.name_and_split = name + '-' + split

        _paths = Dict2Class(config.common_path)
        # _model_para = Dict2Class(config.model_para)

        self.dataset_folder = dataset_folder

        if split in ['train', 'val']:
            self.samples_path = _paths.experiment_data_folder + f'{split}_with_prefetched_ids.json'
        else:
            self.samples_path = data_folder + f'{split}.json'
        self.samples = load_json(self.samples_path)
        self.sid2sample = {sample['id']: sample for sample in self.samples}

        self.papers_path = dataset_folder + 'papers.json'
        self.pid2info = load_json(self.papers_path)

        self.bibref_path = dataset_folder + 'bibref2info.json'
        self.bibref2info = load_json(self.bibref_path)
        self.available_paper_ids = list(self.pid2info.keys())

        self.max_seq_len = max_seq_len
        self.max_num_ref = max_num_ref
        self.max_n_sample = max_n_sample
        self.max_n_positive = max_n_positive
        self.max_n_hard_negative = max_n_hard_negative
        self.max_n_easy_negative = max_n_easy_negative

        self.jaccard_sim = JaccardSim()

        self.bert_tokenizer = tokenizer

        self.paper_batch_size = paper_batch_size

    def __len__(self):
        return len(self.samples)

    def num_papers(self):
        return len(self.pid2info)

    def get_paper_text(self, paper_id) -> str:
        paper_info = self.pid2info[paper_id]
        title = paper_info['title']
        abstract = paper_info['abstract']
        return title + ' [SEP] ' + abstract

    def get_paper_ref_lst(self, paper_id, _max_num_ref=None) -> str:
        paper_info = self.pid2info[paper_id]
        bibref_ids = paper_info['bib_refs']

        if bibref_ids:
            bib_titles = [self.bibref2info[bibref_id]['title'] for bibref_id in bibref_ids if
                          not self.bibref2info[bibref_id]['already_recommended']]
        else:
            bib_titles = []

        if _max_num_ref is None:
            _max_num_ref = self.max_num_ref
        if len(bib_titles) > _max_num_ref:
            bib_titles = bib_titles[:_max_num_ref]
        bib_titles = ' [SEP] '.join(bib_titles)
        return bib_titles

    def __getitem__(self, _):

        class_label_list = []
        irrelevance_level_list = []

        # list of string (title + sep +abstract)
        batch_contents = []

        batch_refs = []
        batch_ids = []

        while True:
            # randomly choose a submission
            idx_of_sample = np.random.choice(len(self.samples))

            # info of the submission
            cur_sample = self.samples[idx_of_sample]
            citing_id = cur_sample["id"]

            batch_contents.append(self.get_paper_text(citing_id))
            batch_refs.append(self.get_paper_ref_lst(citing_id))
            batch_ids.append(citing_id)
            class_label_list.append(idx_of_sample)
            irrelevance_level_list.append(0)

            positive_ids = cur_sample["positive_ids"]
            hard_negative_ids = cur_sample.get("prefetched_ids", [])
            hard_negative_ids = list(set(hard_negative_ids) - set([citing_id] + positive_ids))

            query_text = self.get_paper_text(citing_id)
            if "jaccard_sim_of_positive_ids" not in cur_sample:
                ## get the Jaccard similarity between the query and the positive documents
                jaccard_sim_of_positive_ids = []
                for pos_id in positive_ids:
                    pos_text = self.get_paper_text(pos_id)
                    jaccard_sim_of_positive_ids.append(self.jaccard_sim.compute_sim(query_text, pos_text))
                jaccard_sim_of_positive_ids = np.sort(jaccard_sim_of_positive_ids)
                cur_sample["jaccard_sim_of_positive_ids"] = jaccard_sim_of_positive_ids
            else:
                jaccard_sim_of_positive_ids = cur_sample["jaccard_sim_of_positive_ids"]

            low_thres_jaccard_sim = jaccard_sim_of_positive_ids[int(len(jaccard_sim_of_positive_ids) * 0.15)]
            up_thres_jaccard_sim = jaccard_sim_of_positive_ids[int(len(jaccard_sim_of_positive_ids) * 0.85)]
            avg_thres_jaccard_sim = np.mean(jaccard_sim_of_positive_ids)

            for pos in np.random.choice(len(positive_ids), min(len(positive_ids), self.max_n_positive), replace=False):
                batch_contents.append(self.get_paper_text(positive_ids[pos]))
                batch_refs.append(self.get_paper_ref_lst(positive_ids[pos]))
                batch_ids.append(positive_ids[pos])
                class_label_list.append(idx_of_sample)
                irrelevance_level_list.append(1)

            pos_count = 0
            if self.max_n_hard_negative:
                for pos in np.random.choice(len(hard_negative_ids), len(hard_negative_ids), replace=False):
                    hard_neg_text = self.get_paper_text(hard_negative_ids[pos])
                    hard_negative_jaccard_sim = self.jaccard_sim.compute_sim(query_text, hard_neg_text)

                    if hard_negative_jaccard_sim < low_thres_jaccard_sim:  # low_thres_jaccard_sim  :
                        batch_contents.append(hard_neg_text)
                        batch_refs.append(self.get_paper_ref_lst(hard_negative_ids[pos]))
                        batch_ids.append(hard_negative_ids[pos])
                        class_label_list.append(idx_of_sample)
                        irrelevance_level_list.append(3)
                        pos_count += 1
                        if pos_count >= self.max_n_hard_negative:
                            break
                    elif hard_negative_jaccard_sim > up_thres_jaccard_sim:  # up_thres_jaccard_sim  :
                        batch_contents.append(hard_neg_text)
                        batch_refs.append(self.get_paper_ref_lst(hard_negative_ids[pos]))
                        batch_ids.append(hard_negative_ids[pos])
                        class_label_list.append(idx_of_sample)
                        irrelevance_level_list.append(2)
                        pos_count += 1
                        if pos_count >= self.max_n_hard_negative:
                            break

            self.available_paper_ids = [one for one in self.available_paper_ids if one not in batch_ids]
            for pos in np.random.choice(len(self.available_paper_ids), self.max_n_easy_negative):
                easy_neg_text = self.get_paper_text(self.available_paper_ids[pos])
                batch_contents.append(easy_neg_text)
                batch_refs.append(self.get_paper_ref_lst(self.available_paper_ids[pos]))
                batch_ids.append(self.available_paper_ids[pos])
                class_label_list.append(idx_of_sample)
                irrelevance_level_list.append(4)

            if len(irrelevance_level_list) >= self.max_n_sample:
                break

        outputs = {
            'class_label_list': class_label_list,
            'irrelevance_level_list': irrelevance_level_list,
            'batch_ids': batch_ids,
            'batch_contents': batch_contents,
            'batch_refs': batch_refs
        }

        return outputs

    def paper_batches(self):
        # create batches
        batch_contents = []
        batch_refs = []
        batch_ids = []
        batch_size = self.paper_batch_size
        count = 0
        for paper_id, paper in self.pid2info.items():

            if count % batch_size != 0 or count == 0:
                batch_ids.append(paper_id)
                batch_contents.append(self.get_paper_text(paper_id))
                batch_refs.append(self.get_paper_ref_lst(paper_id))
            else:
                yield batch_contents, batch_refs, batch_ids
                batch_ids = [paper_id]
                batch_contents = [self.get_paper_text(paper_id)]
                batch_refs = [self.get_paper_ref_lst(paper_id)]
            count += 1
        if len(batch_contents) > 0:
            yield batch_contents, batch_refs, batch_ids


class MCLoader:
    def __init__(self, dataset, batch_size, shuffle, worker_init_fn, num_workers=0, drop_last=True, pin_memory=True):
        self.dataloader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     worker_init_fn=worker_init_fn,
                                     num_workers=num_workers,
                                     drop_last=drop_last,
                                     pin_memory=pin_memory)

        def cycle(dataloader):
            while True:
                for x in dataloader:
                    yield x

        self.dataiter = iter(cycle(self.dataloader))

    def get_next(self):
        return next(self.dataiter)


if __name__ == '__main__':
    ds = MCDataset(name='base', split='test')
    mc_loader = MCLoader(dataset=ds, batch_size=1, shuffle=True, worker_init_fn=worker_init_fn)

    batch = mc_loader.get_next()
    print(1)

    from src.modules.models import PLMEncoder
    from src.utils.helper import get_direct_lm_name_and_path

    plm_name, plm_path = get_direct_lm_name_and_path({'model': 'bert'})
    encoder = PLMEncoder(plm_path=plm_path, embed_dim=300,
                         device=torch.device('cuda'),
                         way_ref='atten')
    embedding = encoder(batch)
    for one in ds.paper_batches():
        print(2)
        print(3)
