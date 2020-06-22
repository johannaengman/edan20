import pickle
import regex as re
import sys
import os
import numpy as np

tfidf = {}

def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files


def tokenize(text):
    """
    Uses the letters to bread the text iinto words.
    Returns a list of match objects
    """
    words = re.finditer('\p{L}+', text.lower())
    return words


def text_to_idx(words):
    """
    Builds an index from a list of match objects
    """
    word_idx = {}
    for word in words:
        try:
            word_idx[word.group()].append(word.start())
        except:
            word_idx[word.group()] = [word.start()]
    return word_idx


def tfidf_calc(master, file_length):
    for word in master:
        idf_count = 0
        tf = {}
        for name in file_length:
            if name in master[word]:
                tf[name] = len(master[word][name]) / file_length[name]
                idf_count += 1
            else:
                tf[name] = 0

        nbr_of_files = len(file_length.keys())
        #print(nbr_of_files)

        for name in file_length:
            if name in tfidf.keys():
                tfidf[name][word] = tf[name] * np.log10(nbr_of_files / idf_count)
            else:
                tfidf[name] = {word: tf[name] * np.log10(nbr_of_files / idf_count)}


def print_test():
    test_files = ['bannlyst.txt', 'gosta.txt', 'herrgard.txt', 'jerusalem.txt', 'nils.txt']
    test_words = ['känna', 'gås', 'nils', 'et']
    for f in test_files:
        print(f)
        for w in test_words:
            print(w, tfidf[f][w])


def similarity_clac():
    matrix = np.zeros([9, 9])
    row = 0
    col = 0
    for f1 in tfidf:
        for f2 in tfidf:
            if (f2 != f1):
                x = list(tfidf[f1].values())
                y = list(tfidf[f2].values())
                similarty = np.dot(x, y) / (np.linalg.norm(x) *np.linalg.norm(y))
                matrix[row][col] = similarty
            col += 1
        row += 1
        col = 0
    print(matrix.round(4))
    print(np.argmax(matrix))


if __name__ == '__main__':
    folder_name = sys.argv[1]
    file_names = get_files(folder_name, 'txt')
    print(file_names)
    master = {}
    file_length = {}
    for name in file_names:
        text = open(folder_name + '/' + name).read().lower()
        index = text_to_idx(tokenize(text))
        pickle.dump(index, open(str(name.split('.')[0]) + '.idx', "wb+"))
        file_length[name] = 0
        for key in index:
            file_length[name] += len(index[key])
            if master.get(key):
                master[key][name] = index[key]
            else:
                master[key] = {name: index[key]}
    tfidf_calc(master, file_length)
    similarity_clac()
    #print_test()

