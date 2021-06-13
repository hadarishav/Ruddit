import argparse
import csv
import os
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import BatchEncoding

# Code inspired by https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
GET_ITEM_RETURN = Union[str, BatchEncoding, torch.tensor, int]
os.environ["TOKENIZERS_PARALLELISM"] = "false"
RANDOM_SEED = 12
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class RudditDataset(Dataset):
    """
    Ruddit dataset.
    """

    def __init__(self, comments: np.array, targets: np.array, tokenizer: BertTokenizer, max_len: int, ids: np.array):
        """
        :param comments: Comments from the dataset.
        :param targets: Scores(targets) assigned to the comments.
        :param tokenizer: Bert tokenizer to be used.
        :param max_len: Max length for the BERT tokenizer's encoder.
        :param ids: Ids for keeping track.
        """
        self.reviews = comments
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ids = ids

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item: int) -> Dict[str, GET_ITEM_RETURN]:
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


class RudditLightning(LightningModule):
    """
    Ruddit lightning module.
    """

    def __init__(self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, config: Dict):
        """
        :param df_train: Dataframe containing train data.
        :param df_val: Dataframe containing validation data.
        :param df_test: Dataframe containing test data.
        :param config: Config containing different model parameters.
        """
        super(RudditLightning, self).__init__()
        self.save_hyperparameters()
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.config = config

        self.n_classes = config['num_classes']
        self.max_len = config['max_len']
        self.batch_size = config['batch_size']
        self.max_epochs = config['num_epochs']
        self.PRE_TRAINED_MODEL_NAME = config['PRE_TRAINED_MODEL']
        self.bert = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=config['fc_dropout'])
        extra_dropout = config['bert_dropout']
        for layer in self.bert.encoder.layer:
            layer.attention.self.dropout = torch.nn.Dropout(
                self.bert.config.attention_probs_dropout_prob + extra_dropout)
            layer.output.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob + extra_dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, self.n_classes)
        self.loss = nn.MSELoss().to(self.device)

    # Data Preparation
    def __retrieve_dataset(self, train: bool = True, val: bool = True, test: bool = True) -> RudditDataset:
        """
        Retrieves task specific dataset.

        :param train: Flag. If true, returns train dataloader.
        :param val: Flag. If true, returns val dataloader.
        :param test: Flag. If true, returns test dataloader.
        :return: Train/Val/Test dataloader.
        """
        # return retrieve_data(train, val, test)
        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)

        if train:
            ds = RudditDataset(comments=self.df_train.comment.to_numpy(), targets=self.df_train.Score.to_numpy(),
                               tokenizer=self.tokenizer, max_len=self.max_len, ids=self.df_train.k_id)
        if val:
            ds = RudditDataset(comments=self.df_val.comment.to_numpy(), targets=self.df_val.Score.to_numpy(),
                               tokenizer=self.tokenizer, max_len=self.max_len, ids=self.df_val.k_id)
        if test:
            ds = RudditDataset(comments=self.df_test.comment.to_numpy(), targets=self.df_test.Score.to_numpy(),
                               tokenizer=self.tokenizer, max_len=self.max_len, ids=self.df_test.k_id)
        return ds

    def train_dataloader(self):
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)
        return DataLoader(dataset=self._train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def test_dataloader(self):
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)
        return DataLoader(dataset=self._test_dataset, batch_size=self.batch_size, num_workers=4)

    # Model and Training Preparation
    def forward(self, input_ids: BatchEncoding, attention_mask: BatchEncoding) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[0].mean(dim=1)
        # pooled_output = self.bn(pooled_output)
        output = self.drop(pooled_output)
        return self.out(output)

    def training_step(self,  d: Dict[str, GET_ITEM_RETURN], batch_idx: int) \
            -> Dict[str, Union[np.ndarray, np.ndarray, torch.FloatTensor]]:
        """
        Lightning, training step.

        :param d: Batch containing review text, input ids, attention masks, targets and ids.
        :param batch_idx: Id of the batch (Lightning requirement).
        :return: Dictionary for train_epoch_end.
        """
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

    def test_step(self, d: Dict[str, GET_ITEM_RETURN], batch_idx: int) \
            -> Dict[str, Union[np.ndarray, np.ndarray, torch.FloatTensor]]:
        """
        Lightning, test step.

        :param d: Batch containing review text, input ids, attention masks, targets and ids.
        :param batch_idx: Id of the batch (Lightning requirement).
        :return: Dictionary for test_epoch_end.
        """
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

    def configure_optimizers(self) -> Tuple[List[Union[AdamW, torch.optim.Optimizer]], list]:
        """
        Configure the optimizer and/or scheduler.

        :return: chosen optimizer, scheduler
        """
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], correct_bias=False)
        total_steps = len(self.train_dataloader()) * self.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        return [optimizer], [scheduler]

    def training_epoch_end(self, outputs: list) -> None:
        """
        Called at the end of training epoch.

        :param outputs: List with what is returned in train_step for each batch.
        """
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

    def test_epoch_end(self, outputs: list) -> Dict[str, Any]:
        """
        Called at the end of test epoch.

        :param outputs: List with what is returned in test_step for each batch.
        """
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
    parser.add_argument('--PRE_TRAINED_MODEL', required=True, help='path to pretrained hatebert model',
                        default="/content/drive/MyDrive/hatebert", type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_len', help='max length for BERT tokenizer', default=200, type=int)
    parser.add_argument('--num_classes', help='number of classes', default=1, type=int)
    parser.add_argument('--bert_dropout', help='additional dropout to BERT encoder', default=0.0, type=float)
    parser.add_argument('--fc_dropout', help='dropout to regression head', default=0.0, type=float)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--path_to_train_splits', required=True, help='Path to the folder containing train splits',
                        type=str)
    parser.add_argument('--path_to_test_splits', required=True, help='Path to the folder containing test splits',
                        type=str)
    # parser.add_argument('--wd', default=1e-4, type=float)
    args = parser.parse_args()

    config = {
        'PRE_TRAINED_MODEL': args.PRE_TRAINED_MODEL_NAME,
        'batch_size': args.batch_size,
        'max_len': args.max_len,
        'num_classes': args.num_classes,
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
        df_train = pd.read_csv(f'{args.path_to_train_splits}/train{str(k)}.csv')
        df_test = pd.read_csv(f'{args.path_to_test_splits}/test{str(k)}.csv')
        print('Train:', df_train.shape, ' Test:', df_test.shape)
        start_time = time.time()

        model = RudditLightning(df_train, [], df_test, config)

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
