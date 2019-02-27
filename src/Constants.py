import torch

PAD_IDX = 0
UNK_IDX = 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
GENRES = ['fiction', 'government', 'slate', 'telephone', 'travel']
idx2lab = list(LABELS.keys())
MAXLEN = [26, 15]
max_vocab_size = 40000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")