import argparse
import csv
import os
import time

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

# Code inspired by https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
os.environ["TOKENIZERS_PARALLELISM"] = "false"
RANDOM_SEED = 12
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class AbuseDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len, ids):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ids = ids

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        idx = self.ids[item]

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.float),
            'ids': idx
        }


class AbuseLightning(LightningModule):

    def __init__(self, df_train, df_val, df_test, config):

        super(AbuseLightning, self).__init__()
        self.save_hyperparameters()
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.config = config
        self.n_classes = config['abuse_classes']

        self.max_len = config['max_len']
        self.batch_size = config['batch_size']
        self.max_epochs = config['num_epochs']

        self.PRE_TRAINED_MODEL_NAME = config['PRE_TRAINED_MODEL_NAME']
        self.bert = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=config['fc_dropout'])
        extra_dropout = config['bert_dropout']
        for layer in self.bert.encoder.layer:
            layer.attention.self.dropout = torch.nn.Dropout(
                self.bert.config.attention_probs_dropout_prob + extra_dropout)
            layer.output.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob + extra_dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, self.n_classes)
        self.loss = nn.MSELoss().to(self.device)

    ################################ DATA PREPARATION ############################################

    def __retrieve_dataset(self, train=True, val=True, test=True):

        """ Retrieves task specific dataset """
        # return retrieve_data(train, val, test)
        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)

        if train:
            ds = AbuseDataset(reviews=self.df_train.comment.to_numpy(), targets=self.df_train.Score.to_numpy(),
                              tokenizer=self.tokenizer, max_len=self.max_len, ids=self.df_train.k_id)
        if val:
            ds = AbuseDataset(reviews=self.df_val.comment.to_numpy(), targets=self.df_val.Score.to_numpy(),
                              tokenizer=self.tokenizer, max_len=self.max_len, ids=self.df_val.k_id)
        if test:
            ds = AbuseDataset(reviews=self.df_test.comment.to_numpy(), targets=self.df_test.Score.to_numpy(),
                              tokenizer=self.tokenizer, max_len=self.max_len, ids=self.df_test.k_id)
        return ds

    # @pl.data_loader
    def train_dataloader(self):
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)
        return DataLoader(dataset=self._train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    # @pl.data_loader
    def test_dataloader(self):
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)
        return DataLoader(dataset=self._test_dataset, batch_size=self.batch_size, num_workers=4)

    ################################ MODEL AND TRAINING PREPARATION ############################################

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[0].mean(dim=1)
        # pooled_output = self.bn(pooled_output)
        output = self.drop(pooled_output)

        return self.out(output)

    def training_step(self, d, batch_idx):

        if (self.current_epoch > 5):
            # print('Freezing Bert!')
            for param in self.bert.encoder.parameters():
                param.requires_grad = False

        input_ids = d["input_ids"].to(self.device)
        attention_mask = d["attention_mask"].to(self.device)
        targets = d["targets"].to(self.device)

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.tanh(outputs)
        loss = self.loss(preds.squeeze(dim=1), targets)
        p = preds.squeeze(dim=1).to('cpu').detach().numpy()
        t = targets.to('cpu').detach().numpy()
        loss = loss.type(torch.FloatTensor)

        return {'prediction': p, 'target': t, 'loss': loss}

    def test_step(self, d, batch_idx):

        input_ids = d["input_ids"].to(self.device)
        attention_mask = d["attention_mask"].to(self.device)
        targets = d["targets"].to(self.device)
        ids = d['ids']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.tanh(outputs)
        loss = self.loss(preds.squeeze(dim=1), targets)
        p = preds.squeeze(dim=1).to('cpu').detach().numpy()
        t = targets.to('cpu').detach().numpy()
        loss = loss.type(torch.FloatTensor)

        return {'prediction': p, 'target': t, 'loss': loss, 'ids': ids}

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=self.config['lr'], correct_bias=False)
        total_steps = len(self.train_dataloader()) * self.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        return [optimizer], [scheduler]

    def training_epoch_end(self, outputs):

        # called at the end of the training epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        p = []
        for x in outputs:
            p.extend(x['prediction'])
        t = []
        for x in outputs:
            t.extend(x['target'])
        pear = pearsonr(t, p)
        spear = spearmanr(t, p)
        tau = kendalltau(t, p)
        tensor_pear = torch.tensor(pear[0], device=self.device)
        print(" Train Pearson {}.Train Spearman {}.Train Kendall {} Train Loss {}".format(pear[0], spear[0], tau[0],
                                                                                          avg_loss))
        self.log('train_loss', avg_loss, logger=True)
        self.log('train_pearson', tensor_pear, logger=True)
        # return {'pearson':tensor_pear, 'spearman':spear[0], 'kendall':tau[0], 'loss': avg_loss, 'log': logs}

    def test_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        p = []
        ids = []
        for x in outputs:
            p.extend(x['prediction'])
            ids.extend(x['ids'])
        t = []
        for x in outputs:
            t.extend(x['target'])

        with open('testing_preds.csv', 'a', encoding='utf-8') as f:
            writer = csv.writer(f)
            # writer.writerow(['ID', 'Prediction', 'Target'])
            row = []
            for i, idx in enumerate(ids):
                row.append(idx.item())
                row.append(p[i])
                row.append(t[i])
                writer.writerow(row)
                row = []
        f.close()
        pear = pearsonr(t, p)
        spear = spearmanr(t, p)
        tau = kendalltau(t, p)
        print("Test hparams: ", self.config['lr'], self.config['fc_dropout'], self.config['bert_dropout'])
        print(" Test Pearson {}.Test Spearman {}.Test Kendall {} Test Loss {}".format(pear[0], spear[0], tau[0],
                                                                                      avg_loss))
        return {'test_pearson': pear[0], 'test_spearman': spear[0], 'test_kendall': tau[0], 'test_loss': avg_loss}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Enter args")
    parser.add_argument('--PRE_TRAINED_MODEL_NAME', help='path to pretrained hatebert model',
                        default="/content/drive/MyDrive/hatebert", type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_len', help='max length for BERT tokenizer', default=200, type=int)
    parser.add_argument('--abuse_classes', help='number of classes', default=1, type=int)
    parser.add_argument('--bert_dropout', help='additional dropout to BERT encoder', default=0.0, type=float)
    parser.add_argument('--fc_dropout', help='dropout to regression head', default=0.0, type=float)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    # parser.add_argument('--wd', default=1e-4, type=float)
    args = parser.parse_args()

    config = {
        'PRE_TRAINED_MODEL_NAME': args.PRE_TRAINED_MODEL_NAME,
        'batch_size': args.batch_size,
        'max_len': args.max_len,
        'abuse_classes': args.abuse_classes,
        'bert_dropout': args.bert_dropout,
        'fc_dropout': args.fc_dropout,
        'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        'num_epochs': args.num_epochs,
        'lr': args.lr
    }

    test_loss = []
    test_pearson = []

    for k in range(1, 6):
        print('********************DOING FOLD************************', k)
        df_train = pd.read_csv('train' + str(k) + '.csv')
        df_test = pd.read_csv('test' + str(k) + '.csv')
        print('Train:', df_train.shape, ' Test:', df_test.shape)
        start_time = time.time()

        model = AbuseLightning(df_train, [], df_test, config)

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor='train_loss',
            mode='min')

        trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=0, max_epochs=config['num_epochs'],
                             checkpoint_callback=checkpoint_callback)
        trainer.fit(model)
        test_results = trainer.test(model)
        test_loss.append(test_results[0]['test_loss'])
        test_pearson.append(test_results[0]['test_pearson'])
        end_time = time.time()
        print('********************Finished FOLD************************', k)
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        print('*************Fold took', elapsed_mins, 'minutes and', elapsed_secs, 'seconds')

    print('Average test loss ', np.mean(test_loss))
    print('Standard deviation loss', np.std(test_loss))
    print('Average test pearson ', np.mean(test_pearson))
    print('Standard deviation pearson', np.std(test_pearson))
    print('Losses:', test_loss)
    print('Pearsons:', test_pearson)
