# Transformers started with Machine Translation Datasets
# So for all Transformer Notebooks, lets set up a simple Machine Translation task

# STANDARD LIB
import os
from pathlib import Path
from collections import Counter

# PROGRESS BAR
from tqdm import tqdm

# TORCH IMPORTS
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# TORCHTEXT IMPORTS
from torchtext.vocab import Vocab
from torchtext.datasets import IWSLT2016
from torchtext.data.utils import get_tokenizer

# Utility Functions
def str_2_tensor(str_, tokenizer, vocab):
    encoding = []
    for token in tokenizer(str_):
        # note: 
        # implicitly, if the token is unknown
        # assign to <unk>
        encoding.append(vocab.stoi[token])

    return torch.tensor(encoding, dtype=torch.long)

def dataset_2_loader(device, dataset, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer,
                     PAD_IDX, BOS_IDX, EOS_IDX, batch_size=16):
    
    def generate_batch(data_batch):
        fr_batch, en_batch = [], []
        for (fr_str, en_str) in data_batch:
            fr_tensor = str_2_tensor(fr_str, fr_tokenizer, fr_vocab)
            en_tensor = str_2_tensor(en_str, en_tokenizer, en_vocab)

            fr_batch.append(torch.cat([torch.tensor([BOS_IDX]), fr_tensor, torch.tensor([EOS_IDX])], dim=0))
            en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_tensor, torch.tensor([EOS_IDX])], dim=0))

        # note: pad_sequence is converting the input list into a tensor
        fr_batch = pad_sequence(fr_batch, padding_value=PAD_IDX).to(device)
        en_batch = pad_sequence(en_batch, padding_value=PAD_IDX).to(device)
        return en_batch, fr_batch
    
    # note: we are not able to shuffle a raw iterable dataset
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=generate_batch)
    return loader

# Step 1) Get Dataset
def get_datasets():
    data_root = os.path.join(Path(os.getcwd()).parent.parent.parent, "Datasets/IWSLT2016")
    return IWSLT2016(root=data_root, split=('train', 'valid', 'test'), language_pair=('en', 'fr'), 
                     valid_set='tst2013', test_set='tst2014')

# Step 2) Get Tokenizers
def get_tokenizers():
    fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    return fr_tokenizer, en_tokenizer

# Step 3) Generate Vocabulary from Trainset
def get_vocabs(trainset, en_tokenizer, fr_tokenizer, vocab_max_size=20000):
    en_counter = Counter()
    fr_counter = Counter()
    for (en_sentence, fr_sentence) in trainset:
        en_counter.update(en_tokenizer(en_sentence))
        fr_counter.update(fr_tokenizer(fr_sentence))

        # limit our vocabulary sizes
        if len(en_counter) > vocab_max_size or len(fr_counter) > vocab_max_size:
            break
    
    en_vocab = Vocab(en_counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    fr_vocab = Vocab(fr_counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    return en_vocab, fr_vocab

# Step 4) Get Dataloaders (from raw iterable dataset)
def get_loaders(device):
    trainset, validset, testset = get_datasets()
    fr_tokenizer, en_tokenizer = get_tokenizers()

    # re-get the train data specifically for building the vocab
    # kinda dumb but otherwise we will have already iterated through
    # the entire trainset and next(trainset) will throw an error
    vocab_trainset, _, _ = get_datasets()
    en_vocab, fr_vocab = get_vocabs(vocab_trainset, en_tokenizer, fr_tokenizer)

    # Find Special Tokens
    PAD_IDX = en_vocab['<pad>']
    BOS_IDX = en_vocab['<bos>']
    EOS_IDX = en_vocab['<eos>']

    train_dataloader = dataset_2_loader(device, trainset, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer, 
                                        PAD_IDX, BOS_IDX, EOS_IDX)
    valid_dataloader = dataset_2_loader(device, validset, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer,
                                        PAD_IDX, BOS_IDX, EOS_IDX)
    test_dataloader = dataset_2_loader(device, testset, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer,
                                       PAD_IDX, BOS_IDX, EOS_IDX)
    return train_dataloader, valid_dataloader, test_dataloader, fr_tokenizer, en_tokenizer, en_vocab, fr_vocab

if __name__ == "__main__":
    gpu = torch.device("cuda:0")
    train_dataloader, valid_dataloader, test_dataloader, fr_tokenizer, en_tokenizer, en_vocab, fr_vocab = get_loaders(gpu)
    print(en_vocab.stoi["fhdaklhfkdjah"])
    print(en_vocab["<unk>"])
    for i, (src, trg) in enumerate(train_dataloader):
        print(src.shape)
        break