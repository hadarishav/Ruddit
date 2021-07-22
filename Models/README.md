INSTALLATION
------------

Install the required packages using `pip install -r requirements.txt`


FILES AND INSTRUCTIONS
----------------------

1. In order to create train and test splits for the models, `create_splits.py` should be used. This script requires the dataset to be sorted according to the offensiveness scores and an additional column `k_id` (with row numbers). A sample input file can be found here: `Dataset/sample_input_file.csv`. Comment_ids can be replaced with comments, once extracted (but the script can be tested without replacing the comments as well).

2. **create_splits.py**: This file creates 5-folds of data for cross validation according to sorted stratification: <https://scottclowe.com/2016-03-19-stratified-regression-partitions/>
- To run this file: `python create_splits.py --filename PATH_TO_DATASET --plot FLAG --save_path PATH TO SAVE FOLDS --dataset_range 1`

2. **BiLSTM.py**: This file contains the code for training the BiLSTM model on the dataset.
- Modify line 185 according to your dataset file format
- To run this file on the best setting: `python BiLSTM.py --num_layers 2 --batch_size 32 --num_classes 1 --fc_dropout 0.5 --num_epochs 7 --embedding_dim 300 --hidden_dim 256 --lr 1e-3 --path_to_splits PATH_TO_FOLDER`

3. **BERT.py**: This file contains the code for fine-tuning the BERT base model on the dataset.
- To run this file on the best setting:`python BERT.py --PRE_TRAINED_MODEL_NAME 'bert-base-cased' --batch_size 16 --max_len 200 --num_classes 1 --bert_dropout 0.0 --fc_dropout 0.0 --num_epochs 3 --lr 2e-5 --path_to_train_splits PATH_TO_FOLDER --path_to_test_splits PATH_TO_FOLDER`

4. **HateBERT.py:** This file contains the code for fine-tuning the HateBERT model on the dataset.
- To run this file on the best setting: `python HateBERT.py --PRE_TRAINED_MODEL PATH_TO_MODEL --batch_size 16 --max_len 200 --abuse_classes 1 --bert_dropout 0.0 --fc_dropout 0.0 --num_epochs 3 --lr 2e-5 --path_to_train_splits PATH_TO_FOLDER --path_to_test_splits PATH_TO_FOLDER`

