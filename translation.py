import os
import torch
import spacy
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator
import helper as helper

def translate_sentence(sentence, model, device, max_length=50):
    """
      Translate a German sentence to its English equivalent.

      Args:
          sentence:     German sentence to be translated to English; list or str
          model:        Transformer model used for translation
          device:       device to perform translation on
          max_length:   maximum token length for translation

      Returns:
          src:                  return the tokenized input
          trg_input:            return the input to the decoder before the final output
          trg_output:           return the final translation, shifted right
          attn_probs:           return the attention scores for the decoder heads
          masked_attn_probs:    return the masked attention scores for the decoder heads
    """

    model.eval()
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_trg = load_vocab(spacy_de, spacy_en)
    # tokenize and index the provided string
    if isinstance(sentence, str):
        src = ['<bos>'] + [token.text.lower() for token in spacy_de(sentence)] + ['<eos>']
    else:
        src = ['<bos>'] + sentence + ['<eos>']

    # convert to integers
    src_indexes = [vocab_src[token] for token in src]

    # convert list to tensor
    src_tensor = torch.tensor(src_indexes).int().unsqueeze(0).to(device)

    # set <bos> token for target generation
    trg_indexes = [vocab_trg.get_stoi()['<bos>']]

    # generate new tokens
    for i in range(max_length):
        # convert the list to a tensor
        trg_tensor = torch.tensor(trg_indexes).int().unsqueeze(0).to(device)
        # generate the next token
        with torch.no_grad():

            # generate the logits
            logits = model.forward(src_tensor, trg_tensor)

            # select the newly predicted token
            pred_token = logits.argmax(2)[:, -1].item()

            # if <eos> token or max length, stop generating
            if pred_token == vocab_trg.get_stoi()['<eos>'] or i == (max_length - 1):

                # decoder input
                trg_input = vocab_trg.lookup_tokens(trg_indexes)

                # decoder output
                trg_output = vocab_trg.lookup_tokens(logits.argmax(2).squeeze(0).tolist())

                return src, trg_input, trg_output, model.decoder.attn_probs, model.decoder.masked_attn_probs

            # else, continue generating
            else:
                # add the token
                trg_indexes.append(pred_token)
    return None


def load_tokenizers():
    """
      Load the German and English tokenizers provided by spaCy.
      Returns:
          spacy_de:     German tokenizer
          spacy_en:     English tokenizer
    """
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    print("Loaded English and German tokenizers.")
    return spacy_de, spacy_en


def load_vocab(spacy_de, spacy_en, min_freq: int = 2):
    """
      Args:
          spacy_de:     German tokenizer
          spacy_en:     English tokenizer
          min_freq:     minimum frequency needed to include a word in the vocabulary
      Returns:
          vocab_src:    German vocabulary
          vocab_trg:     English vocabulary
    """

    if not os.path.exists("vocab.pt"):
        # build the German/English vocabulary if it does not exist
        vocab_src, vocab_trg = build_vocabulary(spacy_de, spacy_en, min_freq)
        # save it to a file
        torch.save((vocab_src, vocab_trg), "vocab.pt")
    else:
        # load the vocab if it exists
        vocab_src, vocab_trg = torch.load("vocab.pt")

    print("Finished.\nVocabulary sizes:")
    print("\tSource:", len(vocab_src))
    print("\tTarget:", len(vocab_trg))
    return vocab_src, vocab_trg


def tokenize(text: str, tokenizer):
    """
      Split a string into its tokens using the provided tokenizer.
      Args:
          text:         string
          tokenizer:    tokenizer for the language
      Returns:
          tokenized list of strings
          print(tokenize("A Man with an Orange Hat.", spacy_en))
          # ['a', 'man', 'with', 'an', 'orange', 'hat', '.']
    """
    return [tok.text.lower() for tok in tokenizer.tokenizer(text)]

def build_vocabulary(spacy_de, spacy_en, min_freq: int = 2):
    global vocab_src, vocab_trg
    def tokenize_de(text: str):
        """
          Call the German tokenizer.
          Args:
              text:         string
          Returns:
              tokenized list of strings
        """
        return tokenize(text, spacy_de)

    def tokenize_en(text: str):
        """
          Call the English tokenizer.

          Args:
              text:         string

          Returns:
              tokenized list of strings
        """
        return tokenize(text, spacy_en)

    print("Building German Vocabulary...")

    # load train, val, and test data pipelines
    train = list(helper.load_parallel("data/train.de", "data/train.en"))
    val = list(helper.load_parallel("data/val.de", "data/val.en"))
    test = list(helper.load_parallel("data/test2016.de", "data/test2016.en"))
    #input
    #train = [("Ein Mann mit einem Hut.", "A man with a hat.")]
    #val = [("Eine Frau liest ein Buch.", "A woman is reading a book.")]
    #test = [("Kinder spielen im Park.", "Children are playing in the park.")]
    #output - specials get indices 0-3, in the order passed to `specials=` below
    #(<bos>, <eos>, <pad>, <unk>), then remaining tokens are ordered by frequency
    #(ties broken alphabetically); a word only makes it in at all if it meets min_freq
    #{'<bos>': 0, '<eos>': 1, '<pad>': 2, '<unk>': 3, ...}

    # generate source vocabulary
    try:
        vocab_src = build_vocab_from_iterator(
            yield_tokens(train + val + test, tokenize_de, index=0),  # tokens for each German sentence (index 0)
            min_freq=min_freq,
            specials=["<bos>", "<eos>", "<pad>", "<unk>"],
        )
    except (ValueError, IndexError):
        print('Error happened!')


    print("Building English Vocabulary...")

    # generate target vocabulary
    try:
        vocab_trg = build_vocab_from_iterator(
            yield_tokens(train + val + test, tokenize_en, index=1),  # tokens for each English sentence (index 1)
            min_freq=2,  #
            specials=["<bos>", "<eos>", "<pad>", "<unk>"],
        )
    except (ValueError, IndexError):
        print('Error happened!')

    # set default token for out-of-vocabulary words (OOV)
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_trg.set_default_index(vocab_trg["<unk>"])

    return vocab_src, vocab_trg


def yield_tokens(data_iter, tokenizer, index: int):
    """
      Return the tokens for the appropriate language.
      Args:
          data_iter:    iterable of (german, english) sentence-pair tuples, e.g.:
              [("Ein Mann mit einem Hut.", "A man with a hat."),
               ("Eine Frau liest ein Buch.", "A woman is reading a book.")]
          tokenizer:    tokenizer for the language
          index:        index of the language in the tuple | (de=0, en=1)
      Yields:
          sequences based on index
    """
    for from_tuple in data_iter:
        yield tokenizer(from_tuple[index])