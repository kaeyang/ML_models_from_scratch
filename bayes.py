import sys
import re
import string
import os
import numpy as np
import codecs
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from typing import Sequence

import time

# From scikit learn that got words from:
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])


class defaultintdict(dict):
    """
    Behaves exactly like defaultdict(int) except d['foo'] does NOT
    add 'foo' to dictionary d.
    """

    def __init__(self):
        self._factory = int
        super().__init__()

    def __missing__(self, key):
        return 0


def filelist(root) -> Sequence[str]:
    """Return a fully-qualified list of filenames under root directory; sort names alphabetically."""
    allfiles = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            allfiles.append(os.path.join(path, name))
    return sorted(allfiles)


def get_text(filename: str) -> str:
    """
    Load and return the text of a text file, assuming latin-1 encoding.
    """
    f = open(filename, encoding='latin-1', mode='r')
    s = f.read()
    f.close()
    return s


def words(text: str) -> Sequence[str]:
    """
    Given a string, return a list of words normalized as follows.
    Split the string to make words first by using regex compile() function
    and string.punctuation + '0-9\\r\\t\\n]' to replace all those
    char with a space character.
    Split on space to get word list.
    Ignore words < 3 char long.
    Lowercase all words
    Remove English stop words
    """
    ctrl_chars = '\x00-\x1f'
    regex = re.compile(r'[' + ctrl_chars + string.punctuation + '0-9\r\t\n]')
    # delete stuff but leave at least a space to avoid clumping together
    nopunct = regex.sub(" ", text)
    words = nopunct.split(" ")
    words = [w for w in words if len(w) > 2]  # ignore a, an, to, at, be, ...
    words = [w.lower() for w in words]
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return words


def load_docs(docs_dirname: str) -> Sequence[Sequence]:
    """
    Load all .txt files under docs_dirname and return a list of word lists, one per doc.
    Ignore empty and non ".txt" files.
    """
    docs = []
    file_list = filelist(docs_dirname)
    for file in file_list:
        if file.endswith('.txt') and os.stat(file).st_size > 0:
            docs.append(words(get_text(file)))
    return docs


def vocab(neg: Sequence[Sequence], pos: Sequence[Sequence]) -> dict:
    """
    Given negative and positive lists of word lists in the sentiment_word_list directory, 
    construct a mapping from word to word index.Use index 0 to mean unknown word, '__unknown__'. 
    The real words start from index one.The words should be sorted so the first vocabulary 
    word is index one. The length of the dictionary is |uniquewords|+1 because of "unknown word".
    |V| is the length of the vocabulary including the unknown word slot.

    Sort the unique words in the vocab alphabetically so we standardize which
    word is associated with which word vector index.

    Ex: given neg = [['hi']] and pos=[['mom']], return:
    V = {'__unknown__':0, 'hi':1, 'mom:2}, and so |V| is 3
    """
    V = defaultintdict()
    V['__unknown__'] = 0

    pos_list = [inner for outer in pos for inner in outer]
    neg_list = [inner for outer in neg for inner in outer]
    word_list = pos_list + neg_list
    sort_unique_words = sorted(list(set(word_list)))

    i = 1
    for word in sort_unique_words:
        V[word] = i
        i += 1

    return V


def vectorize(V: dict, docwords: Sequence) -> np.ndarray:
    """
    Return a row vector (based upon V) for docwords with the word counts.
    The first element of the returned vector is the count of unknown words. 
    So |V| is |uniquewords|+1.
    """
    word_count_vec = np.zeros(shape=len(V))
    for word in docwords:
        if word in V.keys():
            word_count_vec[V[word]] += 1
        else:
            word_count_vec[0] += 1

    return word_count_vec


def vectorize_docs(docs: Sequence, V: dict) -> np.ndarray:
    """
    Return a matrix where each row represents a documents word vector.
    Each column represents a single word feature. There are |V|+1
    columns because we leave an extra one for the unknown word in position 0.
    Invoke vector(V,docwords) to vectorize each doc for each row of matrix

    docs: list of word lists, one per doc
    V: Mapping from word to index; e.g., first word -> index 1
    return: numpy 2D matrix with word counts per doc: ndocs x nwords
    """
    nwords = len(V)
    ndocs = len(docs)

    D = np.zeros(shape=(ndocs, nwords))
    count = 0

    for word_list in docs:
        temp = vectorize(V, word_list)
        D[count, :] = temp
        count += 1

    return D


class NaiveBayes:
    """
    This object behaves like a sklearn model with fit(X,y) and predict(X) functions.
    Limited to two classes, 0 and 1 in the y target.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Given 2D word vector matrix X, one row per document, and 1D binary vector y
        train a Naive Bayes classifier. We need to estimate two things, the prior p(c)
        and the likelihood P(w|c). P(w|c) is estimated by
        the number of times w occurs in all documents of class c divided by the
        total words in class c. p(c) is estimated by the number of documents
        in c divided by the total number of documents.

        The first column of X is a column of zeros to represent missing vocab words.
        """
        self.prior_pos = sum(np.where(y == 1, 1, 0))/len(y)
        self.prior_neg = 1 - self.prior_pos

        V_length = X.shape[1]
        y_col_form = y.reshape(-1, 1)
        matrix_table = np.hstack((X, y_col_form))

        pos_docs = matrix_table[matrix_table[:, -1] == 1]
        neg_docs = matrix_table[matrix_table[:, -1] == 0]
        pos_docs = pos_docs[:, :-1]
        neg_docs = neg_docs[:, :-1]

        pos_word_count = np.sum(pos_docs)
        neg_word_count = np.sum(neg_docs)

        pos_numerator = np.sum(pos_docs, axis=0) + 1
        neg_numerator = np.sum(neg_docs, axis=0) + 1

        pos_denom = pos_word_count + V_length  # unknown already in V_length
        neg_denom = neg_word_count + V_length  # unknown already in V_length

        self.word_given_pos_prob = pos_numerator/pos_denom
        self.word_given_neg_prob = neg_numerator/neg_denom

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given 2D word vector matrix X, one row per document, return binary vector
        indicating class 0 or 1 for each row of X.
        """
        y_pred = []
        for row in range(len(X)):
            prob_pos_given_doc = np.log(
                self.prior_pos) + np.sum(np.dot(X[row], np.log(self.word_given_pos_prob)))
            prob_neg_given_doc = np.log(
                self.prior_neg) + np.sum(np.dot(X[row], np.log(self.word_given_neg_prob)))

            if prob_pos_given_doc > prob_neg_given_doc:
                y_pred.append(1)
            else:
                y_pred.append(0)

        return y_pred


def kfold_CV(model, X: np.ndarray, y: np.ndarray, k=4) -> np.ndarray:
    """
    Run k-fold cross validation using model and 2D word vector matrix X and binary
    y class vector. Return a 1D numpy vector of length k with the accuracies, the
    ratios of correctly-identified documents to the total number of documents. Use
    KFold from sklearn to get the splits and loop through the splits to implement
    cross-fold testing. Shuffle the elements when running KFold.
    """
    kf = KFold(k, shuffle=True, random_state=13)
    accuracies = []

    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])

        accuracy = np.sum(y_pred == y[test]) / len(y[test])
        accuracies.append(accuracy)

    return np.array(accuracies)
