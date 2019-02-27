from collections import Counter
import Constants
from dataloader import *
from models import *
import os
import pickle as pkl
from preprocess import *
import torch
from train import *
from test import *
from tools import *

BATCH_SIZE = 64

processed_data_path = "/scratch/yn811/numerical_reasoning_train_val.pkl"
if os.path.exists(processed_data_path):
    print("found existing preprocessed data!")
    train_data, val_data, token2id, id2token = pkl.load(open(processed_data_path, "rb"))
else:
    train_data = pkl.load(open("/scratch/yn811/number_compare_train.pkl", "rb"))
    token_Counter = Counter()
    for idx in range(len(train_data[0])):
        token_Counter.update(train_data[0][idx]+train_data[1][idx])
    token2id, id2token = build_vocab(token_Counter)
    val_data = pkl.load(open("/scratch/yn811/number_compare_val.pkl", "rb"))
    train_data, val_data = token2index_dataset(train_data, token2id), token2index_dataset(val_data, token2id)
    pkl.dump([train_data, val_data, token2id, id2token], open(processed_data_path, "wb"))

pretrained_emb_path = "/scratch/yn811/numerical_reasoning_fasttext_pretrained.pickle"
if os.path.exists(pretrained_emb_path):
    print("found existing loaded pretrained embeddings!")
    pretrained = pkl.load(open(pretrained_emb_path, "rb"))
else:
    pretrained = load_vectors("/scratch/yn811/wiki-news-300d-1M.vec", id2token)
    pkl.dump(pretrained, open(pretrained_emb_path, "wb"))

notPretrained = []
embeddings = [get_pretrain_emb(pretrained, token, notPretrained) for token in id2token]
notPretrained = torch.FloatTensor(np.array(notPretrained)[:, np.newaxis]).to(Constants.DEVICE)


train_dataset = SNLIDataset(train_data)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=collate_func,
                                           shuffle=True)

val_dataset = SNLIDataset(val_data)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=collate_func,
                                           shuffle=True)


test_data = pkl.load(open("/scratch/yn811/number_compare_test_designed.pkl", "rb"))
# test_data = [[test_data[0][idx] for idx in range(10000,20000,1)],
#             [test_data[1][idx] for idx in range(10000,20000,1)],
#             [test_data[2][idx] for idx in range(10000,20000,1)]]
# test_data = [[test_data[0][-idx] for idx in range(10000)],
#             [test_data[1][-idx] for idx in range(10000)],
#             [test_data[2][-idx] for idx in range(10000)]]
test_data = token2index_dataset(test_data, token2id)
test_dataset = SNLIDataset(test_data)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=collate_func,
                                           shuffle=False)


hid_dim = 300
fc_hid_dim = 256
n_layers = 1
embFreeze = True
device = "cuda"
note = "tied"

acc_list = []
for times in range(1):
    print("iteration {}".format(times))
    label = "embFreeze_{}-residual_true-hid_{}-fcHid_{}-nLayers_{}-{}".format(embFreeze, hid_dim, 
                                                                                       fc_hid_dim, n_layers, 
                                                                                       note.replace(' ', '_'))
    
    model = Projector(vocab_size=len(embeddings), 
                      emb_dim=300, hid_dim=hid_dim,
                      n_layers=n_layers, fc_hid_dim=fc_hid_dim, 
                      embeddings=embeddings, device=device).to(device)

    model.embed.weight.requires_grad = not embFreeze

    # model.load_state_dict(torch.load('model' + "-" + label + '.ckpt'))

#     loss_list, val_acc_list = train(model, train_loader, val_loader, 5, label=label)
#     acc_list.append(max(val_acc_list))
#     save_log((loss_list, val_acc_list), label+"-"+str(times))

    model.load_state_dict(torch.load('model' + "-" + label + '.ckpt'))
    print(test_model(test_loader, model))
print(acc_list)