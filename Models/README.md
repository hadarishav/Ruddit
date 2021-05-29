INSTALLATION
------------

Install the required packages using `pip install -r requirements.txt`


FILES AND INSTRUCTIONS
----------------------

1. The dataset should be provided in a csv format with at least the following column headers: [comment, Score, k_id] where 
- comment: is the comment body
- Score: is the offensiveness score of the comment
- k_id: is a serial id, starting from 1, for each comment in the dataset 

2. create_splits.py: This file creates 5-folds of data for cross validation according to sorted stratification: <https://scottclowe.com/2016-03-19-stratified-regression-partitions/>
- To run this file: `python create_splits.py --filename PATH_TO_DATASET --plot FLAG --dataset_range 1`

2. BiLSTM_Ruddit.py: This file contains the code for training the BiLSTM model on the dataset.
- Modify line 185 according to your dataset file format
- To run this file on the best setting: `python BiLSTM.py --num_layers 2 --batch_size 32 --abuse_classes 1 --fc_dropout 0.5 --num_epochs 7 --embedding_dim 300 --hidden_dim 256 --lr 1e-3`

3. BERT_Ruddit.py: This file contains the code for fine-tuning the BERT base model on the dataset.
- To run this file on the best setting:`python BERT.py ----PRE_TRAINED_MODEL_NAME 'bert-base-cased' --batch_size 16 --max_len 200 --abuse_classes 1 --bert_dropout 0.0 --fc_dropout 0.0 --num_epochs 3 --lr 2e-5`

4. HateBERT_Ruddit.py: This file contains the code for fine-tuning the HateBERT model on the dataset.
- To run this file on the best setting: `python HateBERT.py ----PRE_TRAINED_MODEL_NAME PATH_TO_MODEL --batch_size 16 --max_len 200 --abuse_classes 1 --bert_dropout 0.0 --fc_dropout 0.0 --num_epochs 3 --lr 2e-5`

