from collections import defaultdict, Counter

from nltk.corpus import brown

from ourword2vec import Word2Vec
from kmeans import KMeans
from hmm import HMM
from utils import softmax, distance

import numpy as np

# def main():
n_count = 5000
tagged_sents = list(brown.tagged_sents()[:n_count])
brown_sents = list(brown.sents()[:n_count])
tagged_words = [word_tag for sentence in tagged_sents for word_tag in sentence]
words_by_tag = defaultdict(list)
for word, tag in tagged_words:
    words_by_tag[tag].append(word)
words_corpus = [word for word, tag in tagged_words]

print("Generated corpus")

word_vec_length = 100
word2vec = Word2Vec(word_vec_length)

print("Training word2vec")
word2vec.train(words_corpus)
print("Trained word2vec")

print("Calling word2vec on every word")
word_vecs = [word2vec.word2vec(word.lower()) for word in words_corpus]
print("Done")

n_clusters = 10 # random number for now
kmeans = KMeans(n_clusters)
print("Running kmeans")
kmeans.compute(word_vecs)
print("Done")

clusters_corpus = [kmeans.assign_points([word2vec.word2vec(word.lower())])[0] for word in words_corpus]

print("Initializing HMMs")
# word-cluster HMM
word_counter = Counter()
cluster_counter = Counter()

def tiny_float():
    return 0.0001

def defaultdict_tiny_float():
    return defaultdict(tiny_float)

for word in words_corpus:
    word_counter[word] += 1
for cluster in clusters_corpus:
    cluster_counter[cluster] += 1

p_word = defaultdict_tiny_float()
p_cluster = defaultdict_tiny_float()

total_word_counter = sum(word_counter.values())
for word in word_counter:
    p_word[word] = word_counter[word] / total_word_counter

total_cluster_counter = sum(cluster_counter.values())
for cluster in cluster_counter:
    p_cluster[cluster] = cluster_counter[cluster] / total_cluster_counter


count_transition_cluster = defaultdict(Counter)
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
for sentence in brown_sents:
    first_word = sentence[0]
    cluster = kmeans.assign_points([word2vec.word2vec(first_word)])[0]
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
    tag_vec /= len(words_by_tag[tag])
    distances = [distance(tag_vec, k) for k in kmeans.get_k_means()]
    return softmax(distances)[cluster]

count_transition_tag = defaultdict(Counter)
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

print("Initialized")
hmm_word_cluster = HMM(p_initial_cluster, p_transition_cluster, p_word_given_cluster)
hmm_cluster_tag = HMM(p_initial_tag, p_transition_tag, p_cluster_given_tag)
print("Created HMMs, running on input")
words = ['We', 'should', 'receive', 'some', 'portion', 'of', 'these', 'available', 'funds', '.']
print(words)
clusters = hmm_word_cluster.viterbi(words)
print(clusters)
tags = hmm_cluster_tag.viterbi(clusters)
print(tags)

# if __name__ == '__main__':
#    main()
