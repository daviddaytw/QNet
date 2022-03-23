import glob, os

os.environ["CUDA_VISIBLE_DEVICES"]="7,6,5,4"

from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchtext
from torchtext import data
from torchtext.data.utils import get_tokenizer
from torchtext.data.metrics import bleu_score
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator

torch.manual_seed(0)

from config import args
from models.transformer import Seq2Seq
import utils

LOG_NAME = 'Multi30k/transformer-qemb'

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

def log(message):
    with open('logs/' + LOG_NAME + '.log', 'a+') as f:
        f.write(str(message) + '\n')

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def train_one_epoch(model):
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    dataloader = DataLoader(train_iter, batch_size=args.batch_size, collate_fn=collate_batch)
   
    avg_loss = 0

    model.train()
    for src, tgt in tqdm(dataloader, desc='Training'):
        tgt_input = tgt[:, :-1]

        logits = model(src, tgt_input, MAX_SEQ_LEN)

        tgt_out = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

    return avg_loss / len(dataloader)

@torch.no_grad()
def greedy_decode(model, src):
    memory = model(src, None, None, decode=False)
    ys = torch.ones(src.size(0), 1).fill_(BOS_IDX).type(torch.long).to(device)
    for i in range(MAX_SEQ_LEN):
        logits = model(None, ys, ys.size(1), memory=memory)
        ys = torch.cat([ys, torch.argmax(logits, dim=-1)[:, -1:]], dim=1)
    ys = list(ys.cpu().numpy())
    tgt_tokens = []
    for i in ys:
        r = []
        for j in i:
            if j > 3: r.append(j)
            if j == EOS_IDX: break
        tgt_tokens.append(r)
    return [ vocab_transform[TGT_LANGUAGE].lookup_tokens(i) for i in tgt_tokens ]

@torch.no_grad()
def evaluate(model):
    train_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    dataloader = DataLoader(train_iter, batch_size=args.batch_size, collate_fn=collate_batch)
   
    avg_loss = 0
    candidate_corpus = []
    references_corpus = []

    model.eval()
    for src, tgt in tqdm(dataloader, desc='Evaluate'):
        tgt_input = tgt[:, :-1]

        logits = model(src, tgt_input, tgt_input.size(1))

        tgt_out = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        avg_loss += loss.item()
        
        references_corpus.extend([[[vocab_transform[TGT_LANGUAGE].lookup_token(i) for i in l if i > 3]] for l in tgt_out.tolist()])
        candidate_corpus.extend(greedy_decode(model, src))
    print(references_corpus[0], candidate_corpus[0])

    return avg_loss / len(dataloader), bleu_score(candidate_corpus, references_corpus)

if __name__ == '__main__':
    # Init log
    with open('logs/' + LOG_NAME + '.log', 'w') as f:
        f.write('Experiemnt date: ' + time.ctime() + '\n')

    MAX_SEQ_LEN = 64

    # Place-holders
    token_transform = {}
    vocab_transform = {}


    # Create source and target language tokenizer. Make sure to install the dependencies.
    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language=SRC_LANGUAGE)
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language=TGT_LANGUAGE)
        
    # helper function to yield list of tokens
    def yield_tokens(data_iter, language):
        language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

        for data_sample in data_iter:
            yield token_transform[language](data_sample[language_index[language]])

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Training data Iterator
        train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    log(f'Training examples: {len(train_iter)}')
    log('Source Vocab size: ' + str(SRC_VOCAB_SIZE))
    log('Target Vocab size: ' + str(TGT_VOCAB_SIZE))


    # src and tgt language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                                   vocab_transform[ln], #Numericalization
                                                   tensor_transform) # Add BOS/EOS and create tensor

    def collate_batch(batch):
        src_batch = torch.zeros((len(batch), MAX_SEQ_LEN), dtype=torch.long)
        tgt_batch = torch.zeros((len(batch), MAX_SEQ_LEN+1), dtype=torch.long)
        for idx, (src_sample, tgt_sample) in enumerate(batch):
            src_tokens = text_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))
            tgt_tokens = text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n"))
            src_batch[idx] = F.pad(src_tokens, (0, MAX_SEQ_LEN - len(src_tokens)))
            tgt_batch[idx] = F.pad(tgt_tokens, (0, MAX_SEQ_LEN + 1 - len(tgt_tokens)))
        return src_batch.to(device), tgt_batch.to(device)

    model = nn.DataParallel(Seq2Seq(embed_dim=args.embed_dim,
                           num_heads=args.n_heads,
                           num_blocks=args.n_transformer_blocks,
                           src_vocab_size=SRC_VOCAB_SIZE,
                           tgt_vocab_size=TGT_VOCAB_SIZE,
                           max_seq_len=MAX_SEQ_LEN,
                           ffn_dim=args.embed_dim,
                           dropout=args.dropout_rate)).to(device)
    log(f'The model has {utils.count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # training loop
    best_epoch = -1
    best_val = 0
    for iepoch in range(args.n_epochs):
        start_time = time.time()

        log(f"Epoch {iepoch+1}/{args.n_epochs}")

        train_loss = train_one_epoch(model)

        valid_loss, valid_belu = evaluate(model)

        end_time = time.time()

        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        
        print(f'{LOG_NAME} | Epoch: {iepoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. BLEU: {valid_belu*100:.2f}%')
        log(f'Epoch: {iepoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        log(f'\tTrain Loss: {train_loss:.3f}')
        log(f'\t Val. Loss: {valid_loss:.3f} |  Val. BLEU: {valid_belu*100:.2f}%')
        
        if valid_belu > best_val:
            torch.save(model.module.state_dict(), './best_weight.pt')
            best_epoch = iepoch
            best_val = valid_belu
    log(f'Best Epoch: {best_epoch + 1}')