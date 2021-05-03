import os
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

data_root = os.path.join(Path(os.getcwd()).parent.parent.parent, "Datasets/corpora/fr/frwiki-20210120-pages-articles-multistream.xml.bz2")

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Customize pre-tokenization and decoding
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# And then train
trainer = trainers.BpeTrainer(vocab_size=20000, min_frequency=2)
tokenizer.train([data_root], trainer=trainer)

# And Save it
tokenizer.save("byte-level-bpe.tokenizer.json", pretty=True)
