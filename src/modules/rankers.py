import pickle
import numpy as np
import faiss
import logging
import os, sys, inspect
from whoosh.fields import *
from whoosh import scoring, qparser
from whoosh.filedb.filestore import FileStorage, copy_to_ram
from whoosh.index import FileIndex
from whoosh.qparser import MultifieldParser

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, current_dir)

# resolve importing path
from pathlib import Path
import sys

sys.path.append(str(Path().absolute().parent))
sys.path.append(str(Path().absolute().parent.parent))
sys.path.append(str(Path().absolute()))

from src.utils.helper import SentenceTokenizer

logger = logging.getLogger(__name__)

schema = Schema(title=TEXT,
                abstract=TEXT,
                bib_titles=TEXT,
                year=NUMERIC,
                id=ID(stored=True))


class PLMRanker:
    """
      Note: If it requires_precision_conversion = False, this means the document embedding has been pre normalized
      and precision-converted
    """

    def __init__(self, embeddings_path, similarity_type='cos'):
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
        logger.info(f'Embedding loaded from {embeddings_path} for ranking')

        self.similarity_type = similarity_type
        self.index_to_id_mapper = embeddings["index_to_id_mapper"]
        self.id_to_index_mapper = embeddings["id_to_index_mapper"]

        vector_dim = embeddings['embedding'].shape[1]

        ## first normalize the embedding before converting precision, or the precision is float32
        if similarity_type == 'cos':
            self.doc_embeddings = self.normalize_embeddings(embeddings["embedding"])
            self.index_ip = faiss.IndexFlatIP(vector_dim)
        else:
            self.doc_embeddings = embeddings["embedding"]
            self.index_ip = faiss.IndexFlatL2(vector_dim)

        self.index_ip.add(self.doc_embeddings)

        self.encoder = None

    def normalize_embeddings(self, embeddings):
        assert len(embeddings.shape) == 2
        normalized_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        return normalized_embeddings

    def get_top_n_given_embedding(self, n, query_embedding):
        search_results = self.index_ip.search(query_embedding, int(n))
        top_n_indices = search_results[1][0]
        scores = search_results[0][0]
        return [self.index_to_id_mapper[idx] for idx in top_n_indices], scores

    def get_top_n(self, n, qid):

        query_idx = self.id_to_index_mapper[qid]
        query_embedding = self.doc_embeddings[query_idx].reshape(1, -1)

        if self.similarity_type == 'cos':
            query_embedding = self.normalize_embeddings(query_embedding)
        return self.get_top_n_given_embedding(n, query_embedding)


def get_bib_titles(_max_num_ref, paper, bibrefs):
    bibref_ids = paper['bib_refs']
    bib_titles = [bibrefs[bibref_id]['title'] for bibref_id in bibref_ids
                  if not bibrefs[bibref_id]['already_recommended']]
    if len(bib_titles) > _max_num_ref:
        bib_titles = bib_titles[:_max_num_ref]
    bib_titles = ' '.join(bib_titles)

    return bib_titles


class BM25Ranker_whoosh:
    def __init__(self, corpus, index_path, _max_num_ref=50, bibrefs=None):
        super().__init__()
        self.index_path = index_path

        storage = FileStorage(self.index_path, readonly=True)
        self._bm25_index = FileIndex(copy_to_ram(storage), schema=schema)
        self.searcher = self._bm25_index.searcher(weighting=scoring.BM25F)
        self.query_parser = MultifieldParser(['title', 'abstract'],
                                             self._bm25_index.schema, group=qparser.OrGroup)
        self.corpus = corpus

        self._max_num_ref = _max_num_ref
        self.bibrefs = bibrefs

    def get_top_n(self, n, qid):
        paper = self.corpus[qid]

        query_title = paper['title']
        query_abs = paper['abstract']

        title_key_terms = ' '.join([
            t for t, _ in self.searcher.key_terms_from_text('title', query_title)]
        )
        abstract_key_terms = ' '.join([
            t for t, _ in self.searcher.key_terms_from_text('abstract', query_abs)]
        )


        bib_titles = get_bib_titles(_max_num_ref=self._max_num_ref, paper=paper,
                                    bibrefs=self.bibrefs)
        bib_key_terms = ' '.join([
            t for t, _ in self.searcher.key_terms_from_text('bib_titles', bib_titles)]
        )
        query = self.query_parser.parse(title_key_terms + " " + abstract_key_terms + " " + bib_key_terms)

        if n == -1:
            n = len(self.corpus) - 1

        results = self.searcher.search(query, limit=n + 1, optimize=True, scored=True)
        candidate_ids = []
        candidate_scores = []
        for result in results:
            if result['id'] != qid:
                candidate_ids.append(result['id'])
                candidate_scores.append(result.score)

        return candidate_ids, candidate_scores
