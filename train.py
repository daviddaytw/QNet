import os

os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4,5,6"

from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
from torchtext import data
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from config import args, models, datasets
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_log(log_name):
    with open('logs/' + log_name + '.log', 'w') as f:
        f.write('Experiemnt date: ' + time.ctime() + '\n')

def log(log_name, message):
    with open('logs/' + log_name + '.log', 'a+') as f:
        f.write(str(message) + '\n')

def train_model(TextClassifier, dataset, log_name):

    init_log(log_name)
    MAX_SEQ_LEN = args.max_seq_len

    train_iter, test_iter = dataset(split=('train', 'test'))
    train_iter = list(train_iter)
    test_iter = list(test_iter)
    
    label_idx = {}
    for label, _ in train_iter:
        if label not in label_idx:
            label_idx[label] = len(label_idx)
    n_classes = len(label_idx)
    
    log(log_name, f'Training examples: {len(train_iter)}')
    log(log_name, f'Testing examples: {len(test_iter)}')
    

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(utils.yield_tokens(tokenizer, train_iter), specials=["<unk>"], min_freq=10)
    vocab.set_default_index(vocab["<unk>"])
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(label_idx[x])

    def collate_batch(batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = text_pipeline(_text)
            if len(processed_text) > MAX_SEQ_LEN:
                processed_text = processed_text[:MAX_SEQ_LEN]
            for i in range(len(processed_text), MAX_SEQ_LEN):
                processed_text.append(0)
            text_list.append(processed_text)
        label_list = torch.LongTensor(label_list)
        text_list = torch.LongTensor(text_list)
        return label_list.to(device), text_list.to(device)


    train_dataloader = DataLoader(train_iter, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_iter, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    log(log_name, 'Vocab size: ' + str(vocab.__len__()))
    model = TextClassifier(embed_dim=args.embed_dim,
                           num_heads=args.n_heads,
                           num_blocks=args.n_transformer_blocks,
                           num_classes=n_classes,
                           vocab_size=vocab.__len__(),
                           max_seq_len=args.max_seq_len,
                           ffn_dim=args.ffn_dim,
                           n_qubits_transformer=args.n_qubits_transformer,
                           n_qubits_ffn=args.n_qubits_ffn,
                           n_qlayers=args.n_qlayers,
                           dropout=args.dropout_rate)
    net = model.to(device)
#     net = nn.DataParallel(model)
    log(log_name, f'The model has {utils.count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.Adam(lr=args.lr, params=net.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # training loop
    best_epoch = -1
    best_val_acc = 0
    for iepoch in range(args.n_epochs):
        start_time = time.time()

        log(log_name, f"Epoch {iepoch+1}/{args.n_epochs}")

        train_loss, train_acc = utils.train(net, train_dataloader, optimizer, criterion)
        valid_loss, valid_acc = utils.evaluate(net, test_dataloader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        
        print(f'{log_name} | Epoch: {iepoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        log(log_name, f'Epoch: {iepoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        log(log_name, f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        log(log_name, f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        
        if valid_acc > best_val_acc:
            best_epoch = iepoch
            best_val_acc = valid_acc
    log(log_name, f'Best Epoch: {best_epoch}')

if __name__ == '__main__':
    print('Using device:', device)
    for model in models:
        for dataset in datasets:
            train_model(models[model], datasets[dataset], f'{model}_{dataset}')