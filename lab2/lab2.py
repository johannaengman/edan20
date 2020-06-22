import sys
import regex as re
import numpy as np

test_sentance = "det var en gång en katt som hette nils </s>"


def tokenize(text):
    text = re.sub(r'\n', ' ', text)
    #sentance = re.sub(r'(\p{Lu}[^\.]+\. +)', r'<s> \1</s> \n', text)
    sentance = re.sub(r'(\p{Lu}.+?[\.] )', r' <s> \1</s> \n', text)
    #sentance = re.sub(r'\.\s+</s>', r' </s>', sentance)
    #sentance = re.sub(r'([\p{P}])', r'', sentance)
    sentance = re.sub(r'([-!$%^&*()_+|~=`{}\[\]:";?,.])', r'', sentance)
    sentance = sentance.lower()
    #print(sentance)
    return sentance


def count_unigrams(words):
    frequency = {}
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1
    return frequency


def count_bigrams(words):
    bigrams = [tuple(words[inx:inx + 2])
               for inx in range(len(words) - 1)]
    frequencies = {}
    for bigram in bigrams:
        if bigram in frequencies:
            frequencies[bigram] += 1
        else:
            frequencies[bigram] = 1
    return frequencies


def unigram_model(text):
    counts = count_unigrams(text.split())
    nr_words = len(text.split())
    print('Unigram model')
    print('====================================================')
    print('wi', '\t' ,'C(wi)', '\t', '#words', '\t', 'P(wi)')
    print('====================================================')
    prob = 1
    n = 0
    for wi in test_sentance.split():
        n += 1
        probability = counts[wi]/nr_words
        print(wi, '\t', counts[wi], '\t', nr_words, '\t', probability)
        prob = prob*probability
    entropy = -(1/n) * np.log2(prob)
    print('====================================================')
    print("Prob. unigrams: ", prob)
    print("Geometric mean prob.: ", np.power(prob, 1/n))
    print("Entropy rate: ", entropy)
    print("Perplexity: ", np.power(2, entropy))


def bigram_model(text):
    counts_bi = count_bigrams(text.split())
    counts_uni = count_unigrams(text.split())
    number_of_bigram = len(counts_bi.keys())
    print(number_of_bigram)
    nr_words = len(text.split())
    print('Bigram model')
    print('====================================================')
    print('wi', '\t', 'wi+1', '\t', 'Ci,i+1', '\t', 'C(i)', '\t', 'P(wi+1|wi)')
    print('====================================================')
    prob = 1
    n = 0
    prev_word = None
    for wi in test_sentance.split():
        if prev_word:
            n += 1
            bigram = (prev_word, wi)
            if bigram in counts_bi:
                probability = counts_bi[bigram]/ counts_uni[prev_word]
                print(bigram, counts_bi[bigram], counts_uni[prev_word], probability)
                prob *= probability
            else:
                prob_uni = counts_uni[wi] / nr_words
                print(bigram, '0', counts_uni[prev_word], '0.0', "*backoff: ", prob_uni)
                prob *= prob_uni
        else:
            prob_uni = counts_uni[wi] / nr_words
            prob *= prob_uni
        prev_word = wi
    entropy = -(1 / n) * np.log2(prob)
    print('====================================================')
    print("Prob. bigrams: ", prob)
    print("Geometric mean prob.: ", np.power(prob, 1 / n))
    print("Entropy rate: ", entropy)
    print("Perplexity: ", np.power(2, entropy))


if __name__ == '__main__':
    text = sys.stdin.read()
    sentances = tokenize(text)
    unigram_model(sentances)
    bigram_model(sentances)

