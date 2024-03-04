import logging
import pickle
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# resolve importing path
from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))
sys.path.append(str(Path().absolute()))

from src.utils.constants import default_bert_dim

logger = logging.getLogger(__name__)


class CitationParser(nn.Module):

    def __init__(self):
        super(CitationParser, self).__init__()

    def forward(self, content_embedding: Tensor, refs_embeddings: Tensor, return_attn_distribution=False):
        # content: 1 x dim
        # refs: num_ref x dim

        # 1 x num_ref
        sim_scores = torch.matmul(content_embedding, refs_embeddings.T)

        # num_ref x 1
        attn_distrib = F.softmax(sim_scores, dim=1)

        scored_refs = refs_embeddings * attn_distrib.T

        weighted_refs = torch.sum(scored_refs, dim=0).unsqueeze(0)

        if return_attn_distribution:
            return weighted_refs, attn_distrib.T
        else:
            return weighted_refs


class AvgParser(nn.Module):

    def __init__(self):
        super(AvgParser, self).__init__()

    def forward(self, content_embedding: Tensor, refs_embeddings: Tensor):
        # content: 1 x dim
        # refs: num_ref x dim

        # 1 x num_ref
        return torch.mean(refs_embeddings, dim=0).unsqueeze(0)


class PLMEncoder(nn.Module):
    def __init__(self, plm_path, embed_dim, device, way_ref, with_linear=False, max_length=512, ratio_ref=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.device = device
        self.way_ref = way_ref
        self.with_linear = with_linear
        self.max_length = max_length

        self.bert_encoder = AutoModel.from_pretrained(plm_path)
        self.bert_encoder.to(self.device)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(plm_path)

        self.name = plm_path[plm_path.rfind('/') + 1:] + self.way_ref

        if way_ref != 'atten':
            self.citation_parser = AvgParser()
        else:
            self.citation_parser = CitationParser()

        if self.with_linear:
            self.linear = nn.Linear(default_bert_dim, embed_dim)
            self.linear.to(self.device)
        else:
            self.embed_dim = default_bert_dim

        self.ratio_ref = ratio_ref
        if ratio_ref:
            assert 0 <= ratio_ref <= 1

    def forward(self, batch):

        batch_content = [one[0] if type(one) is list else one for one in batch['batch_contents']]
        batch_refs = [one[0] if type(one) is list else one for one in batch['batch_refs']]

        if self.way_ref == 'no':
            # batch_size x 768
            content_embeddings = self.doc_reps_bert(batch_content)
            papers_embeddings = content_embeddings

        elif self.way_ref == 'concat':
            batch_paper = [_content + ' [SEP] ' + _refs for _content, _refs in zip(batch_content, batch_refs)]
            papers_embeddings = self.doc_reps_bert(batch_paper)

        elif self.way_ref in ['atten', 'avg']:
            content_embeddings = self.doc_reps_bert(batch_content)
            assert content_embeddings.shape[0] == len(batch_refs)

            if not self.ratio_ref:
                # batch_size x 2 * embed_dim
                papers_embeddings = torch.zeros([len(batch_refs), 2 * self.embed_dim]).to(self.device)
            else:
                # batch_size x embed_dim
                papers_embeddings = torch.zeros([len(batch_refs), self.embed_dim]).to(self.device)

            for idx, refs_one_paper in enumerate(batch_refs):
                # num_ref_paper x 768
                refs_lst = refs_one_paper.split(' [SEP] ')
                refs_embeddings = self.doc_reps_bert(refs_lst)

                # 1 x 768
                content_embedding = content_embeddings[idx].unsqueeze(0)

                if self.with_linear:
                    refs_embeddings = self.linear(refs_embeddings)
                    content_embedding = self.linear(content_embedding)

                # 1 x 768
                refs_embedding = self.citation_parser(content_embedding, refs_embeddings)

                if not self.ratio_ref:
                    paper_embedding = torch.cat((content_embedding, refs_embedding), dim=1)
                else:
                    paper_embedding = (1 - self.ratio_ref) * content_embedding + self.ratio_ref * refs_embedding

                papers_embeddings[idx] = paper_embedding
        else:
            raise ValueError(f'Selected way_ref={self.way_ref} is not supported!(no concat atten avg)')

        return papers_embeddings

    def doc_reps_bert(self, bert_batch):
        input_ids = self.bert_tokenizer(bert_batch, padding=True, truncation=True,
                                        return_tensors="pt", max_length=self.max_length)

        input_ids = input_ids.to(self.device)

        bert_outputs = self.bert_encoder(**input_ids)

        cls_doc_reps = bert_outputs.last_hidden_state[:, 0, :]  # cls token

        return cls_doc_reps
