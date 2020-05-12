import nltk
from nltk.corpus import stopwords 
import os
import sys
from fnmatch import fnmatch
import string
import math
import operator
import itertools

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    root = os.path.join(".", directory)
    pattern = '*.txt'

    output = dict()

    for path, _, files in os.walk(root):
        for filename in files:
            if fnmatch(filename, pattern):
                file = open(os.path.join(path, filename), 'r')
                output[filename] = file.read()
                file.close()

    return output


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # word_tokenize() puts quotes in pairs
    # --> add these to filter them out more efficiently later
    quotes = "''" + '""' + '``'
    punctuation = string.punctuation + quotes
    stop_words = nltk.corpus.stopwords.words('english') 
    word_tokens = nltk.word_tokenize(document) 

    contents = [
        word.lower() for word in word_tokens 
        if (not word.lower() in stop_words)
        and (not word in punctuation)
    ]

    return contents


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    word_idfs = dict()
    total_docs = len(documents)

    # create a set of all words in all documents
    word_list = []
    for contents in documents.values():
        word_list += contents
    word_set = set(word_list)

    # iterate through each word in word_set
    for word in word_set:
        # create a list of all documents that contain word
        document_list = []
        for file, contents in documents.items():
            if word in contents: 
                document_list.append(file)
            else:
                continue
        # convert doc_list to set to avoid duplicates, and get length
        docs_with_word = len(set(document_list))
        # calculate IDF value for that word and add to word_idfs dictionary
        word_idfs[word] = math.log(total_docs / docs_with_word)

    return word_idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_tfidfs = dict()

    for file, contents in files.items():
        tfidf = 0
        # check each word in the query
        for word in query:
            # if word is in file content
            if word in contents:
                tf = contents.count(word)
                idf = idfs[word]
                tfidf += (tf * idf)
            # skip if file doesn't contain word
            else:
                continue
        file_tfidfs[file] = tfidf

    # sort file_tfidfs dictionary by value
    # reverse=True --> ordered with best match first
    sorted_file_tfidfs = dict(sorted(file_tfidfs.items(), key=operator.itemgetter(1), reverse=True))

    top_n_files = list(sorted_file_tfidfs)[:n]
    return top_n_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_analysis = dict()

    for sentence, words in sentences.items():
        # matching word measure (mwm)
        mwm = 0
        for word in query:
            mwm += idfs[word] if word in words else 0         

        # query term density (qtd) = query word count (qwc) / total word count (twc)
        twc = len(words)
        qwc = 0
        for word in words:
            qwc += 1 if word in query else 0
        qtd = qwc / twc

        sentence_analysis[sentence] = (mwm, qtd)

    sorted_sentence_analysis = dict(sorted(sentence_analysis.items(), key=operator.itemgetter(1), reverse=True))

    top_n_sentences = list(sorted_sentence_analysis)[:n]
    return top_n_sentences


if __name__ == "__main__":
    main()
