from collections import defaultdict, Counter

from nltk.corpus import brown

from word2vec import Word2Vec
from kmeans import KMeans
from hmm import HMM
from utils import softmax, distance

import numpy as np

def main():
    tagged_sents = list(brown.tagged_sents()[2000:2100])
    tagged_words = [word_tag for sentence in tagged_sents for word_tag in sentence]
    words_by_tag = defaultdict(list)
    for word, tag in tagged_words:
        words_by_tag[tag].append(word)
    words_corpus = list(brown.words())

    word_vec_length = 2000
    word2vec = Word2Vec(word_vec_length)
    word2vec.train(words_corpus)

    word_vecs = [word2vec.word2vec(word) for word in words_corpus]

    n_clusters = 10 # random number for now
    kmeans = KMeans(n_clusters)
    kmeans.compute(word_vecs)

    # word-cluster HMM
    p_word = {}
    p_cluster = {}
    clusters_corpus = [kmeans.assign_points([word2vec.word2vec(word)])[0] for word in words_corpus]

    def tiny_float():
        return 0.000000001

    def defaultdict_tiny_float():
        return defaultdict(tiny_float)

    count_transition_cluster = Counter()
    prev_cluster = None
    for cluster in clusters_corpus:
        if prev_cluster is not None:
            count_transition_cluster[prev_cluster][cluster] += 1
        prev_cluster = cluster

    p_transition_cluster = defaultdict(defaultdict_tiny_float)
    for prev_cluster in count_transition_cluster:
        next_cluster_counter = count_transition_cluster[prev_cluster]
        total_next_clusters = sum(next_cluster_counter.values())
        for next_cluster in next_cluster_counter:
            p_transition_cluster[prev_cluster][next_cluster] = next_cluster_counter[next_cluster]/total_next_clusters

    count_initial_cluster = Counter()
    for sentence in brown.sents():
        first_word = sentence[0]
        cluster = kmeans.assign_points([first_word])[0]
        count_initial_cluster[cluster] += 1

    p_initial_cluster = defaultdict_tiny_float()
    total_initial_clusters = sum(count_initial_cluster.values())
    for cluster in count_initial_cluster:
        p_initial_cluster[cluster] = count_initial_cluster[cluster] / total_initial_clusters

    def p_cluster_given_word(cluster, word):
        """
        cluster is a number 0 <= cluster <= k
        """
        vec = word2vec.word2vec(word)
        distances = [distance(vec, k) for k in kmeans.get_k_means()]
        return softmax(distances)[cluster]

    def p_word_given_cluster(word, cluster):
        return p_cluster_given_word(cluster, word) * p_word[word] / p_cluster[cluster]

    # cluster-tag HMM
    def p_cluster_given_tag(cluster, tag):
        tag_vec = np.zeros(word_vec_length)
        for word in words_by_tag[tag]:
            tag_vec += word2vec.word2vec(word)
        distances = [distance(tag_vec, k) for k in kmeans.get_k_means()]
        return softmax(distances)[cluster]

    count_transition_tag = Counter()
    prev_tag = None
    for word, tag in tagged_words:
        if prev_tag is not None:
            count_transition_tag[prev_tag][tag] += 1
        prev_tag = tag

    p_transition_tag = defaultdict(defaultdict_tiny_float)
    for prev_tag in count_transition_tag:
        next_tag_counter = count_transition_tag[prev_tag]
        total_next_tags = sum(next_tag_counter.values())
        for next_tag in next_tag_counter:
            p_transition_tag[prev_tag][next_tag] = next_tag_counter[next_tag]/total_next_tags

    count_initial_tag = Counter()
    for sentence in tagged_sents:
        first_word, first_tag = sentence[0]
        count_initial_tag[first_tag] += 1

    p_initial_tag = defaultdict_tiny_float()
    total_initial_tags = sum(count_initial_tag.values())
    for tag in count_initial_tag:
        p_initial_tag[tag] = count_initial_tag[tag] / total_initial_tags

    hmm_word_cluster = HMM(p_initial_cluster, p_transition_cluster, p_word_given_cluster)
    hmm_cluster_tag = HMM(p_initial_tag, p_transition_tag, p_cluster_given_tag)

    words = []
    clusters = hmm_word_cluster.viterbi(words)
    tags = hmm_cluster_tag.viterbi(clusters)

if __name__ == '__main__':
    main()
