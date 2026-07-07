import torch
import torch.nn as nn
from embeddings import Embeddings
from encoder_layer import Encoder
from decoder_layer import Decoder
from transformer import Transformer
from positional_encoding import PositionalEncoding
import translation as translation
from torch.nn.functional import pad
from torch import Tensor
# visualization packages
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def make_model(device,
               src_vocab,
               trg_vocab,
               n_layers: int = 3,
               d_model: int = 512,
               d_ffn: int = 2048,
               n_heads: int = 8,
               dropout: float = 0.1,
               max_length: int = 5000):
    """
      Construct a model when provided parameters.
      Args:
          src_vocab:    source vocabulary
          trg_vocab:    target vocabulary
          n_layers:     number of stacked layers in the encoder and in the decoder
          d_model:      dimension of embeddings
          d_ffn:        dimension of feed-forward network
          n_heads:      number of heads
          dropout:      probability of dropout occurring
          max_length:   maximum sequence length for positional encodings
      Returns:
          Transformer model based on hyperparameters
      """
    # create source embedding matrix
    src_embed = Embeddings(len(src_vocab), d_model)

    # create target embedding matrix
    trg_embed = Embeddings(len(trg_vocab), d_model)

    # create a positional encoding matrix
    pos_enc = PositionalEncoding(d_model, dropout, max_length)

    # create the encoder
    encoder = Encoder(d_model, n_layers, n_heads, d_ffn, dropout)

    # create the decoder
    decoder = Decoder(len(trg_vocab), d_model, n_layers, n_heads, d_ffn, dropout)

    # create the Transformer model
    model = Transformer(
        encoder,
        decoder,
        nn.Sequential(src_embed, pos_enc),
        nn.Sequential(trg_embed, pos_enc),
        src_pad_idx=src_vocab.get_stoi()["<pad>"],
        trg_pad_idx=trg_vocab.get_stoi()["<pad>"],
        device=device
    )

    # initialize parameters with Xavier/Glorot
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def train(model, iterator, optimizer, criterion, clip):
    """
      Train the model on the given data.
      Args:
          model:        Transformer model to be trained
          iterator:     data to be trained on
          optimizer:    optimizer for updating parameters
          criterion:    loss function for updating parameters
          clip:         value to help prevent exploding gradients
      Returns:
          loss for the epoch
    """

    # set the model to training mode
    model.train()

    epoch_loss = 0

    # loop through each batch in the iterator
    for i, batch in enumerate(iterator):
        # set the source and target batches
        src, trg = batch

        # zero the gradients
        optimizer.zero_grad()

        # logits for each output
        logits = model(src, trg[:, :-1])

        # expected output
        expected_output = trg[:, 1:]

        # calculate the loss
        loss = criterion(logits.contiguous().view(-1, logits.shape[-1]),
                         expected_output.contiguous().view(-1))

        # backpropagation
        loss.backward()

        # clip the weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update the weights
        optimizer.step()

        # update the loss
        epoch_loss += loss.item()

    # return the average loss for the epoch
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    """
      Evaluate the model on the given data.
      Args:
          model:        Transformer model to be trained
          iterator:     data to be evaluated
          criterion:    loss function for assessing outputs
      Returns:
          loss for the data
    """

    # set the model to evaluation mode
    model.eval()

    epoch_loss = 0

    # evaluate without updating gradients
    with torch.no_grad():
        # loop through each batch in the iterator
        for i, batch in enumerate(iterator):
            # set the source and target batches
            src, trg = batch

            # logits for each output
            logits = model(src, trg[:, :-1])

            # expected output
            expected_output = trg[:, 1:]

            # calculate the loss
            loss = criterion(logits.contiguous().view(-1, logits.shape[-1]),
                             expected_output.contiguous().view(-1))

            # update the loss
            epoch_loss += loss.item()

    # return the average loss for the epoch
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs


def compute_metrics(model, iterator):
    """
      Generate predictions for the provided iterator.
      Args:
          model:        Transformer model to be trained
          iterator:     data to be evaluated
      Returns:
          predictions:  list of predictions, which are tokenized strings
          labels:       list of expected output, which are tokenized strings
    """

    # set the model to evaluation mode
    model.eval()

    predictions = []
    labels = []

    # load tokenizers/vocab once instead of on every batch
    spacy_de, spacy_en = translation.load_tokenizers()
    vocab_src, vocab_trg = translation.load_vocab(spacy_de, spacy_en)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # translate_sentence adds its own <bos>/<eos>, so strip specials out of the raw batch tensor first
    src_special_idxs = {vocab_src.get_stoi()[tok] for tok in ('<bos>', '<eos>', '<pad>')}

    # evaluate without updating gradients
    with torch.no_grad():
        # loop through each batch in the iterator
        for i, batch in enumerate(iterator):
            # set the source and target batches
            src, trg = batch

            # translate_sentence works on a single sentence, so loop over each row in the batch
            for src_ids, trg_ids in zip(src, trg):
                src_tokens = vocab_src.lookup_tokens(
                    [idx for idx in src_ids.tolist() if idx not in src_special_idxs])

                _, trg_input, trg_output, attn_probs, masked_attn_probs = translation.translate_sentence(
                    src_tokens, model, device)

                # prediction | remove <eos> token
                predictions.append(trg_output[:-1])

                # expected output | add extra dim for calculation
                labels.append([vocab_trg.lookup_tokens(trg_ids.tolist())])

    return predictions, labels

def data_process(raw_data):
  """
    Process raw sentences by tokenizing and converting to integers based on
    the vocabulary.
    Args:
        raw_data:     German-English sentence pairs
    Returns:
        data:         tokenized data converted to index based on vocabulary
  """

  data = []
  spacy_de, spacy_en = translation.load_tokenizers()
  vocab_src, vocab_trg = translation.load_vocab(spacy_de, spacy_en)
  # loop through each sentence pair
  for (raw_de, raw_en) in raw_data:
    # tokenize the sentence and convert each word to an integers
    de_tensor_ = torch.tensor([vocab_src[token.text.lower()] for token in spacy_de.tokenizer(raw_de)], dtype=torch.long)
    en_tensor_ = torch.tensor([vocab_trg[token.text.lower()] for token in spacy_en.tokenizer(raw_en)], dtype=torch.long)

    # append tensor representations
    data.append((de_tensor_, en_tensor_))
  return data


def generate_batch(data_batch):
    """
      Process indexed-sequences by adding <bos>, <eos>, and <pad> tokens.
      Args:
          data_batch:     German-English indexed-sentence pairs
      Returns:
          two batches:    one for German and one for English
    """
    de_batch, en_batch = [], []
    spacy_de, spacy_en = translation.load_tokenizers()
    vocab_src, vocab_trg = translation.load_vocab(spacy_de, spacy_en);
    BOS_IDX = vocab_trg['<bos>']
    EOS_IDX = vocab_trg['<eos>']
    PAD_IDX = vocab_trg['<pad>']
    MAX_PADDING = 20
    BATCH_SIZE = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');

    # for each sentence
    for (de_item, en_item) in data_batch:
        # add <bos> and <eos> indices before and after the sentence
        de_temp = torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0).to(device)
        en_temp = torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0).to(device)

        # add padding
        de_batch.append(pad(de_temp, (0,  # dimension to pad
                                      MAX_PADDING - len(de_temp),  # amount of padding to add
                                      ), value=PAD_IDX, ))

        # add padding
        en_batch.append(pad(en_temp, (0,  # dimension to pad
                                      MAX_PADDING - len(en_temp),  # amount of padding to add
                                      ),
                            value=PAD_IDX, ))

    return torch.stack(de_batch), torch.stack(en_batch)


def display_attention(
        sentence: list,
        translation: list,
        attention: Tensor,
        n_heads: int = 8,
        n_rows: int = 4,
        n_cols: int = 2):
    """
      Display the attention matrix for each head of a sequence.
      Args:
          sentence:     German sentence to be translated to English; list
          translation:  English sentence predicted by the model
          attention:    attention scores for the heads
          n_heads:      number of heads
          n_rows:       number of rows
          n_cols:       number of columns
    """
    # ensure the number of rows and columns are equal to the number of heads
    assert n_rows * n_cols == n_heads

    # figure size
    fig = plt.figure(figsize=(15, 25))

    # visualize each head
    for i in range(n_heads):

        # create a plot
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        # select the respective head and make it a numpy array for plotting
        _attention = attention.squeeze(0)[i, :, :].cpu().detach().numpy()

        # plot the matrix
        cax = ax.matshow(_attention, cmap='bone')

        # set the size of the labels
        ax.tick_params(labelsize=12)

        # set the indices for the tick marks
        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(translation)))

        # if the provided sequences are sentences or indices
        if isinstance(sentence[0], str):
            ax.set_xticklabels([t.lower() for t in sentence], rotation=45)
            ax.set_yticklabels(translation)
        elif isinstance(sentence[0], int):
            ax.set_xticklabels(sentence)
            ax.set_yticklabels(translation)

    plt.show()

def load_parallel(de_path, en_path):
    with open(de_path, encoding="utf-8") as f_de, open(en_path, encoding="utf-8") as f_en:
        for de, en in zip(f_de, f_en):
            yield de.strip(), en.strip()