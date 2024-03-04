from pathlib import Path
import os
import time
import sys

# get absolute path of the project
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent) + '/'

data_folder = PROJECT_ROOT + 'data/'
base_folder = data_folder + 'base/'
extended_folder = data_folder + 'extended/'
experiments_folder = PROJECT_ROOT + 'experiments/'
scripts_folder = PROJECT_ROOT + 'src/'

if os.path.exists(PROJECT_ROOT + 'model_hub/'):
    model_hub_folder = PROJECT_ROOT + 'model_hub/'
else:
    model_hub_folder = 'online or set your path'

default_bert_dim = 768

val_paper_embedding_file = 'val_paper_embedding.pkl'
test_paper_embedding_file = 'test_paper_embedding.pkl'
train_paper_embedding_file = 'train_paper_embedding.pkl'

default_K_list = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
metrics_names = ['map', 'mrr', 'ndcg', 'recall', 'precision', 'f1']

model2online_path = {
    'specter': 'allenai/specter',
    'scincl': 'malteos/scincl',
    'citebert': 'copenlu/citebert',
    'scibert': 'allenai/scibert_scivocab_uncased',
    'bert': 'bert-base-uncased',
    'linkbert': 'michiyasunaga/LinkBERT-base'
}

if __name__ == '__main__':
    print(PROJECT_ROOT)
