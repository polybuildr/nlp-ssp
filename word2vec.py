from nltk.corpus import brown
import nltk
import numpy as np

class Word2Vec:
    def __init__(self, vec_length=2000):
        """
        vec_length: The desired length of the vector representation for each word
        """
        self.vec_length = vec_length
        self.cap_limit = 32767

    def train(self, words):
        """
        Build an internal representation of the word corpus to use for calculating vectors.
        """
        self.corpus = map(lambda word: word.lower(), words)
        self.words = nltk.FreqDist(self.corpus)
        self.words = map(lambda tup: tup[0], self.words.most_common())
        self.topK = self.words[:min(len(self.words), self.vec_length)]
        self.topKdict = dict()
        for word in self.topK:
            self.topKdict[word] = 1

        self.wordToIdx = dict()
        for i in xrange(len(self.words)):
            self.wordToIdx[self.words[i]] = i

        self.cMatrix = np.zeros([len(self.words), self.vec_length], dtype='int16')
        for i in xrange(len(self.corpus)):
            word = self.corpus[i]
            if not self.wordToIdx.has_key(word):
                continue
            idx = self.wordToIdx[word]
            for j in xrange(i - 2, i + 3):
                if j < 0 or j >= len(self.corpus):
                    continue
                other = self.corpus[j]
                if not self.topKdict.has_key(other):
                    continue
                count = self.cMatrix[idx][self.wordToIdx[other]]
                if count < self.cap_limit:
                    count += 1
                self.cMatrix[idx][self.wordToIdx[other]] = count

    def word2vec(self, word):
        """
        Return the word vector of the given word.
        """
        return self.cMatrix[self.wordToIdx[word.lower()]]