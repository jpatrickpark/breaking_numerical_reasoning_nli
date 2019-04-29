import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
import time
import os
import csv
import sys
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import json
version = "bert-base-cased"

data_dir = "/scratch/yn811/snli_1.0/"
val_file = "augmented_dev.jsonl"
train_file = "augmented_train.jsonl"
additional_train_file = "snli_1.0_train.jsonl"
additional_val_file = "snli_1.0_dev.jsonl"
max_seq_len = 96
batch_size = 64
n_epochs = 2
learning_rate = 1e-5
validate_every = 1000
print_every = 500
max_n_steps = 2000000
n_classes = 3
n_train_steps = -1
save_model_name = "3-classes-2-epoch-mixed"

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def evaluate(model, dataloader, batch_size=32):
    model.eval()
    eval_loss, eval_accuracy = 0., 0.
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy}
    return result

def train(model, save_model_name):
    print("***** Running training *****")
    print("  Num examples = %d" % len(train_examples))
    print("  Batch size = %d" % batch_size)
    #   print("  Num steps = %d" % n_train_steps)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    start_time = time.time()
  
    for epoch_n in range(1, int(n_epochs) + 1):
        print("Epoch %d" % epoch_n)
        tr_loss = 0.
        nb_tr_examples, nb_tr_steps = 0, 0
        best_acc = 0
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if (step + 1) % print_every == 0:
                print("\tStep %d:\ttrain loss %.3f" % (step + 1, tr_loss / nb_tr_examples))
                print("\t\tTrained on %d examples in %.3f" % (nb_tr_examples, time.time() - start_time))
                start_time = time.time()
            if (step + 1) % validate_every == 0:
                print("\tValidating...")
                results = evaluate(model, eval_dataloader, batch_size=batch_size)
                print("\t\tdev accuracy: %.3f" % results["eval_accuracy"])
                if results["eval_accuracy"] > best_acc:
                    torch.save(model.state_dict(), 'model' + "-" + save_model_name + '.ckpt')
                    best_acc = results["eval_accuracy"]

            if step >= max_n_steps:
                return

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        
    
def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace(
        '-LRB-', '(').replace('-RRB-', ')').lower()
        
        
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    
    @classmethod
    def _read_json(cls, input_file, skip_no_majority=True, limit=None, snli=True):
        """Reads a txt file."""
        if input_file is None:
            return None
        for i, line in enumerate(open(input_file)):
            if limit and i > limit:
                break
            data = json.loads(line)
            label = data['gold_label']
            if label == '-':
                continue
            if 'sentence1_binary_parse' in data.keys():
                s1 = extract_tokens_from_binary_parse(data['sentence1_binary_parse'])
                s2 = extract_tokens_from_binary_parse(data['sentence2_binary_parse'])
            else:
                s1 = extract_tokens_from_binary_parse(data['sentence1'])
                s2 = extract_tokens_from_binary_parse(data['sentence2'])
            if skip_no_majority and label == '-':
                continue
            yield (s1, s2, label)


class SNLIProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir, file_name, additional_train_file):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, file_name)), \
            self._read_json(os.path.join(data_dir, additional_train_file)), "train")
    

    def get_dev_examples(self, data_dir, file_name, additional_val_file):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, file_name)), \
            self._read_json(os.path.join(data_dir, additional_val_file)), "dev")
#         return self._create_examples(
#             self._read_json(os.path.join(data_dir, "snli_1.0_dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ['entailment', 'contradiction', 'neutral']
    
    
    def _create_examples(self, lines, additional_lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if additional_lines is not None:
            lines = list(lines) + list(additional_lines)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
          

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, n_classes=3):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : min(i, n_classes-1) for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if len(input_ids) > max_seq_len:
            continue
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length, "{}".format(len(input_ids))
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

# Prepare model
tokenizer = BertTokenizer.from_pretrained(version)
model = BertForSequenceClassification.from_pretrained(version, num_labels = n_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = model.to(device)

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate, warmup=0.1, t_total=n_train_steps)


processor = SNLIProcessor()
label_list = processor.get_labels()

train_examples = processor.get_train_examples(data_dir, train_file, additional_train_file)
train_features = convert_examples_to_features(train_examples, label_list, max_seq_len, tokenizer, n_classes)

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

eval_examples = processor.get_dev_examples(data_dir, val_file, additional_val_file)
eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_len, tokenizer, n_classes)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

n_train_steps = int(len(train_examples) / batch_size) * n_epochs
# model.load_state_dict(torch.load('model' + "-" + "2-classes-2-epoch" + '.ckpt'))

train(model, save_model_name)
results = evaluate(model, eval_dataloader, batch_size=batch_size)
print("Final dev acc: %.3f" % results["eval_accuracy"])