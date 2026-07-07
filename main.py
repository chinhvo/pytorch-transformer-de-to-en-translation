import helper as helper
import torch
import torch.nn as nn

import time
import math
from torch.utils.data import DataLoader
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.functional import to_map_style_dataset
import torchtext.datasets as datasets
import translation as translation

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')
MAX_PADDING = 20
BATCH_SIZE = 128

train_data_raw = list(helper.load_parallel("data/train.de", "data/train.en"))
val_data_raw = list(helper.load_parallel("data/val.de", "data/val.en"))
test_data_raw = list(helper.load_parallel("data/test2016.de", "data/test2016.en"))

# processed data
train_data = helper.data_process(train_data_raw)
val_data = helper.data_process(val_data_raw)
test_data = helper.data_process(test_data_raw)

train_iter = DataLoader(to_map_style_dataset(train_data), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, collate_fn=helper.generate_batch)

valid_iter = DataLoader(to_map_style_dataset(val_data), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, collate_fn=helper.generate_batch)

test_iter = DataLoader(to_map_style_dataset(test_data), batch_size=BATCH_SIZE,
                       shuffle=True, drop_last=True, collate_fn=helper.generate_batch)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spacy_de, spacy_en = translation.load_tokenizers()
vocab_src, vocab_trg = translation.load_vocab(spacy_de, spacy_en)

BOS_IDX = vocab_trg['<bos>']
EOS_IDX = vocab_trg['<eos>']
PAD_IDX = vocab_trg['<pad>']

model =  helper.make_model(device, vocab_src, vocab_trg,
                   n_layers=3, n_heads=8, d_model=256,
                   d_ffn=512, max_length=50)
model.to(device)

LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

# loop through each epoch

for epoch in range(N_EPOCHS):

    start_time = time.time()

    # calculate the train loss and update the parameters
    train_loss = helper.train(model, train_iter, optimizer, criterion, CLIP)

    # calculate the loss on the validation set
    valid_loss = helper.evaluate(model, valid_iter, criterion)

    end_time = time.time()

    # calculate how long the epoch took
    epoch_mins, epoch_secs = helper.epoch_time(start_time, end_time)

    # save the model when it performs better than the previous run
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# load the weights
model.load_state_dict(torch.load('transformer-model.pt'))

# calculate the loss on the test set
#test_loss = helper.evaluate(model, test_iter, criterion)

#print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')


# 'a woman with a large purse is walking by a gate'
src = ['eine', 'frau', 'mit', 'einer', 'großen', 'geldbörse', 'geht', 'an', 'einem', 'tor', 'vorbei', '.']

src, trg_input, trg_output, attn_probs, masked_attn_probs = translation.translate_sentence(src, model, device)

print(f'source = {src}')
print(f'target input = {trg_input}')
print(f'target output = {trg_output}')

helper.display_attention(trg_input, trg_input, masked_attn_probs)
