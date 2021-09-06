Ruddit: Norms of Offensiveness for English Reddit Comments
----------------------------------------------------------

Ruddit is a dataset of English language Reddit comments that has fine-grained, real-valued scores between -1 (maximally supportive) and 1 (maximally offensive). Data sampling procedure, annotation, and analysis have been discussed in detail in the accompanying paper. We provide the comment IDs, post IDs and not the bodies, in accordance to the GDPR regulations. The comments and post bodies can be extracted from any Reddit API using the IDs provided. 

The paper can be found here: [Ruddit: Norms of Offensiveness for English Reddit Comments](https://aclanthology.org/2021.acl-long.210/)

If you use our work, please cite us:

```bash

@inproceedings{hada-etal-2021-ruddit,
    title = "Ruddit: {N}orms of Offensiveness for {E}nglish {R}eddit Comments",
    author = "Hada, Rishav  and
      Sudhir, Sohi  and
      Mishra, Pushkar  and
      Yannakoudakis, Helen  and
      Mohammad, Saif M.  and
      Shutova, Ekaterina",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.210",
    doi = "10.18653/v1/2021.acl-long.210",
    pages = "2700--2717",
    abstract = "On social media platforms, hateful and offensive language negatively impact the mental well-being of users and the participation of people from diverse backgrounds. Automatic methods to detect offensive language have largely relied on datasets with categorical labels. However, comments can vary in their degree of offensiveness. We create the first dataset of English language Reddit comments that has fine-grained, real-valued scores between -1 (maximally supportive) and 1 (maximally offensive). The dataset was annotated using Best{--}Worst Scaling, a form of comparative annotation that has been shown to alleviate known biases of using rating scales. We show that the method produces highly reliable offensiveness scores. Finally, we evaluate the ability of widely-used neural models to predict offensiveness scores on this new dataset.",
}

```

This repository contains the dataset (with only comment and post ids) for Ruddit and the code for creating it's variants and for training the models, as reported in the paper. The repository is structured as follows:

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

* `index.html` contains the code for the annotation task interface. The task interface can be viewed [here](https://hadarishav.github.io/Ruddit/).
