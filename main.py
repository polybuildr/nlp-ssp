from kmeans import KMeans
from hmm import HMM

def main():
    pass
    tagged_words = []
    words_corpus = []

    n_clusters = 10 # random number for now
    kmeans = KMeans(n_clusters)
    kmeans.compute(words_corpus)

    # word-cluster HMM
    p_word = {}
    p_cluster = {}

    p_cluster_given_word = None # softmax
    p_word_given_cluster = None # joint probability formula

    p_transition_cluster = None # count
    p_initial_cluster = None # count

    # cluster-tag HMM
    p_cluster_given_tag = None # softmax
    p_transition_tag = None # count from tagged data
    p_initial_tag = None # count from tagged data

    hmm_word_cluster = HMM(p_initial_cluster, p_transition_cluster, p_word_given_cluster)
    hmm_cluster_tag = HMM(p_initial_tag, p_transition_tag, p_cluster_given_tag)

    words = []
    clusters = hmm_word_cluster.viterbi(words)
    tags = hmm_cluster_tag.viterbi(clusters)

if __name__ == '__main__':
    main()
