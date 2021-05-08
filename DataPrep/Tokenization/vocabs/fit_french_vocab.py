import os
from pathlib import Path
from pprint import pprint

from tokenizers import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Whitespace

from transformers import PreTrainedTokenizerFast

def get_french_vocab(model_name):
    root = Path(os.getcwd()).parent.parent.parent
    french_corpus = "Datasets/corpora/fr/text"
    fr_corpus_path = os.path.join(root, french_corpus)
    files = []
    for dir_ in os.listdir(fr_corpus_path):
        fr_corpus_dir = os.path.join(fr_corpus_path, dir_)
        for text_file in os.listdir(fr_corpus_dir):
            text_file = os.path.join(fr_corpus_dir, text_file)
            files.append(text_file)

    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
    tokenizer.pre_tokenizer = Whitespace()
    
    tokenizer.train(files,
                    vocab_size=20000,
                    min_frequency=2,
                    show_progress=True,
                    special_tokens=["<sos>", "<pad>", "<eos>", "<unk>"])
    
    print(tokenizer.encode("c'est la meilleure des phrases françaises").tokens)
    tokenizer.save(model_name)

def load_french_vocab(model_name):
    #tokenizer = PreTrainedTokenizerFast(tokenizer_object=model_name)
    #print(tokenizer.encode("c'est la meilleure des phrases françaises").tokens)
    tokenizer = ByteLevelBPETokenizer("wiki_fr_tokenizer.json", add_prefix_space=True)
    print(tokenizer.encode("c'est la meilleure des phrases françaises").tokens)


if __name__ == "__main__":
    model_name = "wiki_fr_tokenizer.json"
    #get_french_vocab(model_name)
    load_french_vocab(model_name)
