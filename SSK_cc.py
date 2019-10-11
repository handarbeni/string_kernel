import initialize_dataset as ds
from sklearn import svm
import numpy as np
from functools import lru_cache
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(15000)


class SVM:
    def __init__(self, name, train, test, train_label, test_label, n):
        self.name = name
        self.subseq_length = n
        self.lambda_decay = 0.5
        self.train_set = train
        self.test_set = test
        self.train_labels = train_label
        self.test_labels = test_label
        self.text_clf = svm.SVC(kernel='precomputed')
        self.S = self.find_sub_sequences_of_n(self.train_set, n)
        print("number n",n)
        print("subseq", len(self.S))
        self.n = n

    def ngrams(self, text, n):
        if (len(text) < n):
            return []

        text_length = len(text)
        text_list = (text_length - n + 1) * [0]
        text_list[0] = text[0:n]
        for i in range(1, text_length - n + 1):
            text_list[i] = text_list[i - 1][1:] + text[i + n - 1]
        return text_list

    def find_sub_sequences_of_n(self, train_set, n):
        list_ngram = []
        for text in train_set:
            list_ngram += self.ngrams(text, n)

        count = Counter(list_ngram)
        count = {k: v for k, v in count.items() if v > 10}
        cuple = sorted(count.items(), key=lambda x: x[1], reverse=True)
        return [c[0] for c in cuple]

    @lru_cache(maxsize=None)
    def _K(self, n, s, t):
        """
        K_n(s,t) in the original article; recursive function
        :param n: length of subsequence
        :type n: int
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :return: float value for similarity between s and t
        """
        if min(len(s), len(t)) < n:
            return 0
        else:
            part_sum = 0
            for j in range(1, len(t)):
                if t[j] == s[-1]:
                   part_sum += self._K1(n - 1, s[:-1], t[:j])
            result = self._K(n, s[:-1], t) + self.lambda_decay ** 2 * part_sum
            return result

    @lru_cache(maxsize=None)
    def _K1(self, n, s, t):
        """
        K'_n(s,t) in the original article; auxiliary intermediate function; recursive function
        :param n: length of subsequence
        :type n: int
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :return: intermediate float value
        """
        if n == 0:
            return 1
        elif min(len(s), len(t)) < n:
            return 0
        else:
            part_sum = 0
            for j in range(1, len(t)):
                if t[j] == s[-1]:
                    part_sum += self._K1(n - 1, s[:-1], t[:j]) * (self.lambda_decay ** (len(t) - (j + 1) + 2))
            result = self.lambda_decay * self._K1(n, s[:-1], t) + part_sum
            return result

    def train(self, train, label):
        print("Training model " + self.name)
        self.text_clf = svm.SVC(kernel='precomputed')
        gram = self.gramMatrix(train, train)
        self.text_clf.fit(gram, label)
        print("Traning done")

    def predict(self, train, test):
        gram = self.gramMatrix(test, train)
        Y = self.text_clf.predict(gram)
        return Y

    def gramMatrix(self, X, Y):
        len_X = len(X)
        len_Y = len(Y)

        gram = np.zeros((len(X), len(Y)))
        normalizing_kernel = {}
        normalizing_kernel[0] = {}
        normalizing_kernel[1] = {}

        # store K(s,s) values in dictionary to avoid recalculations
        for i in range(len(X)):
            normalizing_kernel[0][i] = self._K(self.subseq_length, X[i], X[i])

        for i in range(len(Y)):
            normalizing_kernel[1][i] = self._K(self.subseq_length, Y[i], Y[i])

        # Calculating the kernel value for the documents
        for i in range(len_X):
            for j in range(len_Y):
                if gram[i][j] == 0:
                    resultKernel = self._gram_matrix_element(X[i], Y[j], normalizing_kernel[0][i],
                                                             normalizing_kernel[1][j])
                    gram[i][j] = resultKernel
                    # Exploiting the symmetric property of the gram matrix
                    if (j < len_X) and (i < len_Y) and (i != j):
                        gram[j][i] = resultKernel
        return gram

    def _gram_matrix_element(self, s, t, sdkvalue1, sdkvalue2):
        """
        Helper function
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :param sdkvalue1: K(s,s) from the article
        :type sdkvalue1: float
        :param sdkvalue2: K(t,t) from the article
        :type sdkvalue2: float
        :return: value for the (i, j) element from Gram matrix
        """
        if s == t:
            return 1
        else:
            try:
                return self._K(self.subseq_length, s, t) / \
                       (sdkvalue1 * sdkvalue2) ** 0.5
            except ZeroDivisionError:
                print("Maximal subsequence length is less or equal to documents' minimal length."
                      "You should decrease it")
                sys.exit(2)

    def evaluatePrediction(self, prediction, test_labels):
        correct_predictions = 0.0
        for i in range(len(test_labels)):
            if prediction[i] == test_labels[i]:
                correct_predictions += 1

        print("accuracy", correct_predictions/len(test_labels))


train_data,train_label,test_data,test_label = ds.reduce_data()
print(len(train_data))
print(len(train_label))
print(len(test_data))
print(len(test_label))

SVM_SSK = SVM("SVM_SSK", train_data[:10], test_data[:4], train_label[:10], test_label[:4], 3)
print(SVM_SSK.subseq_length)
SVM_SSK.train(train_data[:10], train_label[:10])
predicted_output = SVM_SSK.predict(train_data[:10], test_data[:4])
SVM_SSK.evaluatePrediction(predicted_output, test_label[:4])

