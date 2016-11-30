from collections import defaultdict

from nltk.corpus import brown

from word2vec import Word2Vec
from kmeans import KMeans
from hmm import HMM
from utils import softmax

import numpy as np

def main():
    tagged_words = list(brown.tagged_words()[10000:11200]) # 1200 words, approx 0.1% of brown
    words_by_tag = defaultdict(list)
    for word, tag in tagged_words:
        words_by_tag[tag].append(word)
    words_corpus = list(brown.words())

    word_vec_length = 20000
    word2vec = Word2Vec(word_vec_length)
    word2vec.train(words_corpus)

    word_vecs = [word2vec.word2vec(word) for word in words_corpus]

    n_clusters = 10 # random number for now
    kmeans = KMeans(n_clusters)
    kmeans.compute(word_vecs)

    # word-cluster HMM
    p_word = {}
    p_cluster = {}

    def distance(v1, v2):
        return np.linalg.norm(np.array(v1) - np.array(v2))

    def p_cluster_given_word(cluster, word):
        """
        cluster is a number 0 <= cluster <= k
        """
        vec = word2vec.word2vec(word)
        distances = [distance(vec, k) for k in kmeans.get_k_means()]
        return softmax(distances)[cluster]

    def p_word_given_cluster(word, cluster):
        return p_cluster_given_word(cluster, word) * p_word[word] / p_cluster[cluster]

    p_transition_cluster = None # count from words_corpus
    p_initial_cluster = None # count from words_corpus

    # cluster-tag HMM
    def p_cluster_given_tag(cluster, tag):
        tag_vec = np.zeros(word_vec_length)
        for word in words_by_tag[tag]:
            tag_vec += word2vec.word2vec(word)
        distances = [distance(tag_vec, k) for k in kmeans.get_k_means()]
        return softmax(distances)[cluster]

    p_transition_tag = None # count from tagged data
    p_initial_tag = None # count from tagged data

    hmm_word_cluster = HMM(p_initial_cluster, p_transition_cluster, p_word_given_cluster)
    hmm_cluster_tag = HMM(p_initial_tag, p_transition_tag, p_cluster_given_tag)

    words = []
    clusters = hmm_word_cluster.viterbi(words)
    tags = hmm_cluster_tag.viterbi(clusters)

if __name__ == '__main__':
    main()
