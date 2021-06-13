import argparse
import csv
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))


def tokenize_comments(ruddit: str) -> List[List[str]]:
    """
    Reads ruddit(csv), tokenizes the comments and removes the stop words
    from them and returns a list containing the tokenized comments.

    :param ruddit: Path to the csv containing the dataset.
    :return tokenized_comments: Returns a list of list where each list contains
    tokenized words per comment.
    """
    tokenizer = RegexpTokenizer(r'\w+')
    with open(ruddit, 'r', encoding='utf-8') as cfile:

        tokenized_comments = []
        reader = csv.reader(cfile)
        for i, line in enumerate(reader):

            # Ignore the column headers
            if (i == 0):
                continue

            # Tokenize the lower-cased comment
            word_tokens = tokenizer.tokenize(line[0].lower())
            filtered_sentence = [w for w in word_tokens if w not in stop_words]
            tokenized_comments.append(filtered_sentence)

        cfile.close()

    return tokenized_comments


def create_dataset_without_swearwords(ruddit: str, cursing_lexicon: str,
                                      tokenized_comments: List[List[str]],
                                      variant_save_path: str) -> None:
    """
    Removes the swearwords from the tokenized comments and writes
    the 'no-swearing' version of the dataset to csv.

    :param ruddit: Path to the csv containing the dataset.
    :param cursing_lexicon: Path to 'cursing_lexicon.txt'.
    :param tokenized_comments: List containing tokenized comments.
    :param variant_save_path: Path to save the new dataset variant.
    """

    def getswearwords(cursing_lexicon: str) -> List[str]:
        """
      Extracts swear words from the txt file.

      :param cursing_lexicon: Path to 'cursing_lexicon.txt'.
      :return list containing swearwords.
      """
        swearwords = []
        f1 = open(cursing_lexicon, 'r')
        lines = f1.readlines()
        for line in lines:
            swearwords.append(line.strip())
        return swearwords

    # Get swearwords from the lexicon txt
    swearwords = getswearwords(cursing_lexicon)

    variant_save_path = 'no_swearing.csv' if variant_save_path is None else variant_save_path
    with open(ruddit, 'r', encoding='utf-8') as dfile, \
            open(variant_save_path, 'w', encoding='utf-8') as nfile:

        # File reader and writer
        reader = csv.reader(dfile)
        writer = csv.writer(nfile)

        # Counter for number of comments containing swear words
        with_swear_words = 0

        for i, line in enumerate(reader):

            # Copy the column headers
            if (i == 0):
                writer.writerow(line)
                continue

            tc = tokenized_comments[i - 1]
            flag = 0
            for word in tc:

                if (word in swearwords):
                    # Update counter
                    with_swear_words += 1
                    flag = 1
                    break

            # Check if comment contained swear word(s)
            if (flag != 1):
                writer.writerow(line)

        dfile.close()
        nfile.close()
        print(f'Number of comments to be removed: {with_swear_words}')


def create_dataset_without_identityterms(ruddit: str, identity_file: str,
                                         variant_save_path: str) -> None:
    """
    Replaces the identity terms from the lemmatized comments 
    and writes the 'identity-agnostic' version of the dataset to csv.

    :param ruddit: Path to the csv containing the dataset.
    :param identity_file: Path to 'identity_terms.txt'.
    :param variant_save_path: Path to save the new dataset variant.
    """
    def getidentityterms(identity_file):
        """
      Extracts identity terms from the txt file.

      :param identity_file: Path to 'identity_terms.txt'.
      :return list containing identity terms.
      """
        identityterms = []
        f1 = open(identity_file, 'r')
        lines = f1.readlines()
        for line in lines:
            identityterms.append(line.strip())
        return identityterms

    # Get identity terms from the identity txt
    identity_terms = getidentityterms(identity_file)

    variant_save_path = 'identity_agnostic.csv' if variant_save_path is None else variant_save_path
    with open(ruddit, 'r', encoding='utf-8') as ifile, \
            open(variant_save_path, 'w', encoding='utf-8') as ofile:

        # File reader and writer
        reader = csv.reader(ifile)
        writer = csv.writer(ofile)

        # Counter for number of comments with identity terms
        with_identity_terms = 0

        tokenizer = RegexpTokenizer(r'\w+')
        lemmatizer = WordNetLemmatizer()

        for i, line in enumerate(reader):

            # Copy the column headers
            if (i == 0):
                writer.writerow(line)
                continue

            # Tokenize and lemmatize the comments
            word_tokens = tokenizer.tokenize(line[0])
            lemmas = [lemmatizer.lemmatize(t) for t in word_tokens]

            # Check if there are words to be replaced
            words_to_replace = []
            for index, word in enumerate(lemmas):
                if (word.lower() in identity_terms):
                    words_to_replace.append(word_tokens[index])

            # Replace the identity terms with group
            if (len(words_to_replace) > 0):
                with_identity_terms += 1
                comment = line[0]
                for j, w in enumerate(words_to_replace):
                    comment = comment.replace(w, 'group')
                line[0] = comment
                writer.writerow(line)

            else:
                writer.writerow(line)

        ofile.close()
        print(f'Number of comments with identity terms: {with_identity_terms}')


def create_dataset_with_reduced_range(ruddit: str,
                                      variant_save_path: str) -> None:
    """
    Keeps comments having scores from -0.5 and 0.5 and creates
    the 'reduced-range' version of the dataset to csv.

    :param ruddit: Path to the csv containing the dataset.
    :param variant_save_path: Path to save the new dataset variant.
    """
    variant_save_path = 'reduced_range.csv' if variant_save_path is None else variant_save_path
    with open(ruddit, 'r', encoding='utf-8') as ifile, \
            open(variant_save_path, 'w', encoding='utf-8') as ofile:

        # File reader and writer
        reader = csv.reader(ifile)
        writer = csv.writer(ofile)

        for i, line in enumerate(reader):

            # Copy the column headers
            if (i == 0):
                writer.writerow(line)
                continue

            score = float(line[-1])
            if (score >= -0.5 and score <= 0.5):
                writer.writerow(line)

        ifile.close()
        ofile.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create Ruddit dataset variant")
    parser.add_argument('--dataset_type', help="Enter 1 for no-swearing version, 2 for identity_agnostic "
                        "and 3 for reduced range", default=2, type=int)
    parser.add_argument('--ruddit', help='Path to Ruddit dataset', default='Ruddit.csv', type=str)
    parser.add_argument('--cursing_lexicon', help="Path to swearing lexicon", default='cursing_lexicon.txt', type=str)
    parser.add_argument('--identity_file', help="Path to swearing lexicon", default='identityterms_group.txt', type=str)
    parser.add_argument('--variant_save_path', help="Path to save the variant", type=str)
    args = parser.parse_args()

    # Call the appropriate function
    if(args.dataset_type not in [1, 2, 3]):
        print('Invalid value for dataset type, assigning value 3.')
    if (args.dataset_type == 1):
        tokenized_comments = tokenize_comments(args.ruddit)
        create_dataset_without_swearwords(ruddit=args.ruddit, cursing_lexicon=args.cursing_lexicon,
                                          tokenized_comments=tokenized_comments,
                                          variant_save_path=args.variant_save_path)
    elif (args.dataset_type == 2):
        create_dataset_without_identityterms(ruddit=args.ruddit, identity_file=args.identity_file,
                                             variant_save_path=args.variant_save_path)
    else:
        create_dataset_with_reduced_range(ruddit=args.ruddit, variant_save_path=args.variant_save_path)
