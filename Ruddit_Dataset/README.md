Ruddit: Norms of Offensiveness for English Reddit Comments
----------------------------------------------------------

Ruddit is a dataset of English language Reddit comments that has fine-grained, real-valued scores between -1 (maximally supportive) and 1 (maximally offensive).

Data sampling procedure, annotation, and analysis have been discussed in detail in the accompanying paper.

We provide the comment IDs, post IDs and not the bodies, in accordance to the GDPR regulations. The comments and post bodies can be extracted from any Reddit API using the IDs provided. 

FILES AND FORMAT
----------------

1. **Ruddit.csv**: This is the main dataset file with 6000 comments and their respective scores. 
It has three columns (separated by commas):
- comment_id: The ID of the comment for which the offensiveness score is provided.
- post_id: The ID of the main post on which the given comment was made. 
- offensiveness_score: The degree of offensiveness score of the comment.

2. **node_dictionary.npy**: This is a Python NumPy format file that contains the entire comment thread for each unique post ID included in Ruddit.
- This is a dictionary with post_id as the key and a python anytree `Node` (that provides the entire comment thread for that post) as the value.

3. **load_node_dictionary.py**: Sample code to access `node_dictionary.npy` and print the tree structure of comment thread for a given post ID.
- It uses the Python Anytree library. 
- Command to install the Anytree library: `pip install anytree`
- Please read the Anytree library documentation to know more about the functionality it provides <https://anytree.readthedocs.io/en/latest/>. 

4. **Thread_structure.txt**: Printed comment thread tree structure for the 372 unique post IDs included in Ruddit.

5. **create_dataset_variants.py**: This file contains code to create the different dataset variants (as specified in the paper).
- It uses the Python nltk library (`pip install nltk`)