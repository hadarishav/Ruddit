Ruddit Norms of Offensiveness for English Reddit Comments
---------------------------------------------------------------

* This repository contains the dataset (with only comment and post ids) for Ruddit and the code for creating it's variants and for training the models, as reported in the paper.

The repository is structured as follows:

```bash
.
├── Dataset
│   ├── create_dataset_variants.py
│   ├── identityterms_group.txt
│   ├── load_node_dictionary.py
│   ├── node_dictionary.npy
│   ├── README.md
│   ├── Ruddit.csv
│   ├── Ruddit_individual_annotations.csv
│   └── Thread_structure.txt
├── index.html
├── LICENSE
├── Models
│   ├── BERT.py
│   ├── BiLSTM.py
│   ├── create_splits.py
│   ├── HateBERT.py
│   └── README.md
├── README.md
└── requirements.txt

```

* **Dataset** contains the Ruddit dataset, the individual annotations, the identity terms txt file and the code for creating the dataset variants from the original Ruddit dataset. (For the *cursing lexicon*, you can email the authors of the paper or drop an email to us).

* **Models** contains the code for creating the splits for each dataset variant and the code for creating training the **BERT**, **HateBERT** and **BiLSTM** models.

* Each folder has their separate Readme for further instructions.

* `requirements.txt` lists down the requirements for running the code on the repository. The requirements can be downloaded using : `pip install -r requirements.txt`