import argparse
import csv
import time

import numpy as np
import torch
from torchtext import data
from torch import nn
from scipy.stats import pearsonr

# Code adapted from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/
# 1%20-%20Simple%20Sentiment%20Analysis.ipynb
RANDOM_SEED = 12
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        hidden = hidden[0]

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)


def train(model, iterator, optimizer, criterion):
    epoch_loss = []

    model.train()

    p, t = [], []

    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.c

        predictions = model(text, text_lengths)

        predictions = torch.tanh(predictions)

        loss = criterion(predictions.squeeze(1), batch.s)

        p.extend(predictions.squeeze(dim=1).to('cpu').detach().numpy())
        t.extend(batch.s.to('cpu').detach().numpy())

        loss.backward()

        optimizer.step()

        epoch_loss.append(loss.item())

    pear = pearsonr(np.array(t), np.array(p))[0]

    return np.mean(epoch_loss), pear


def evaluate(model, iterator, criterion):
    epoch_loss = []

    model.eval()

    p, t = [], []
    ids = []

    with torch.no_grad():

        for batch in iterator:
            text, text_lengths = batch.c

            ids.extend(batch.k)

            predictions = model(text, text_lengths)

            predictions = torch.tanh(predictions)

            loss = criterion(predictions.squeeze(1), batch.s)

            p.extend(predictions.squeeze(dim=1).to('cpu').detach().numpy())
            t.extend(batch.s.to('cpu').detach().numpy())

            epoch_loss.append(loss.item())

        # This csv is created for keeping track of the predictions made by the model on the test set
        with open('testing_preds_glove.csv', 'a', encoding='utf-8') as f:
            writer = csv.writer(f)
            # writer.writerow(['ID', 'Prediction', 'Target'])
            row = []
            for i, idx in enumerate(ids):
                row.append(idx.item())
                row.append(p[i])
                row.append(t[i])
                writer.writerow(row)
                row = []

        pear = pearsonr(np.array(t), np.array(p))[0]

    return np.mean(epoch_loss), pear


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enter args")
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--abuse_classes', default=1, type=int)
    parser.add_argument('--fc_dropout', default=0.5, type=float)
    parser.add_argument('--num_epochs', default=7, type=int)
    parser.add_argument('--embedding_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    args = parser.parse_args()

    config = {
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'abuse_classes': args.abuse_classes,
        'fc_dropout': args.fc_dropout,
        'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'hidden_dim': args.hidden_dim,
        'embedding_dim': args.embedding_dim
    }

    losses = []
    pearson = []
    for num in range(5):

        train_fname = 'train' + str(num + 1) + '.csv'
        test_fname = 'test' + str(num + 1) + '.csv'

        comment = data.Field(tokenize='spacy', include_lengths=True, lower=True)
        score = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
        k_id = data.Field(sequential=False, use_vocab=False)

        fields = [('c', comment), (None, None), (None, None), (None, None), (None, None), ('s', score), ('k', k_id)]
        # Modify this according to your csv. k_id was a list of indices added for creating folds for crossvalidation
        # fields = [('c', comment), ('s', Score), ('k', k_id)]

        train_data, test_data = data.TabularDataset.splits(
            path='/content/',
            train=train_fname,
            test=test_fname,
            format='csv',
            fields=fields,
            skip_header=True
        )

        comment.build_vocab(train_data, vectors="glove.6B.300d",
                            unk_init=torch.Tensor.normal_)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        BATCH_SIZE = config['batch_size']

        train_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, test_data),
            sort_key=lambda x: len(x.c),  # sort by s attribute (quote)
            sort_within_batch=True,
            batch_size=BATCH_SIZE,
            device=device)

        INPUT_DIM = len(comment.vocab)
        EMBEDDING_DIM = config['embedding_dim']
        HIDDEN_DIM = config['hidden_dim']
        OUTPUT_DIM = config['abuse_classes']
        N_LAYERS = config['num_layers']
        BIDIRECTIONAL = True
        DROPOUT = config['fc_dropout']
        PAD_IDX = comment.vocab.stoi[comment.pad_token]

        model = RNN(INPUT_DIM,
                    EMBEDDING_DIM,
                    HIDDEN_DIM,
                    OUTPUT_DIM,
                    N_LAYERS,
                    BIDIRECTIONAL,
                    DROPOUT,
                    PAD_IDX)

        pretrained_embeddings = comment.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)
        UNK_IDX = comment.vocab.stoi[comment.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        model = model.to(device)
        criterion = criterion.to(device)

        N_EPOCHS = config['num_epochs']
        print(f'The model has {count_parameters(model):,} trainable parameters')

        for epoch in range(N_EPOCHS):
            start_time = time.time()

            train_loss, train_pear = train(model, train_iterator, optimizer, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Pear: {train_pear:.2f}')

        test_loss, test_pear = evaluate(model, test_iterator, criterion)
        losses.append(test_loss)
        pearson.append(test_pear)

        print(f'Test Loss: {test_loss:.3f} | Test Pear: {test_pear:.2f}')
        print('Ended fold', num + 1)

    print('num_layers', config['num_layers'], 'hidden_dim', config['hidden_dim'])
    print('Losses:', losses)
    print('Avg test loss:', np.mean(losses))
    print('Standard deviation loss:', np.std(losses))
    print('Pearson:', pearson)
    print('Avg test pearson:', np.mean(pearson))
    print('Standard deviation pearson:', np.std(pearson))
