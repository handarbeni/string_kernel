import initialize_dataset as ds
from sklearn import svm
import numpy as np
from backports.functools_lru_cache import lru_cache
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import sys
from time import time
import gc
sys.setrecursionlimit(15000)


class SVM:
    def __init__(self, name, n):
        self.name = name
        self.subseq_length = n
        self.lambda_decay = 0.5
        self.text_clf = svm.SVC(kernel='precomputed')
        self.n = n


    @lru_cache(maxsize=50000000)
    def _K2_LP(self, n, m, s, t):
        """
        K''_n(s,t) in the original article; auxiliary intermediate function; recursive function
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
            if s[-1] == t[-1]:
                return self.lambda_decay * (self._K2_LP(n, m-1, s, t[:-1]) +
                                            self.lambda_decay * self._K1_LP(n - 1, m-2, s[:-1], t[:-1]))
            else:
                u = len(s)+len(t[:-1])
                return pow(self.lambda_decay, u) * self._K2_LP(n, m-u, s, t[:-1])

    @lru_cache(maxsize=50000000)
    def _K1_LP(self, n, m, s, t):
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
        if m < 2*n:
            return 0
        else:
            if n == 0:
                return 1
            elif min(len(s), len(t)) < n:
                return 0
            else:
                result = self._K2_LP(n, m, s, t) + self.lambda_decay * self._K1_LP(n, m-1, s[:-1], t)
                return result


    @lru_cache(maxsize=50000000)
    def _K_LP(self, n, m, s, t):
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
                    part_sum += self._K1_LP(n - 1, m-2, s[:-1], t[:j])
            result = self._K_LP(n, m, s[:-1], t) + self.lambda_decay ** 2 * part_sum
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

        m = len_X + len_Y

        # store K(s,s) values in dictionary to avoid recalculations
        for i in range(len(X)):
            normalizing_kernel[0][i] = self._K_LP(self.subseq_length, m, X[i], X[i])

        if X != Y:
            for i in range(len(Y)):
                normalizing_kernel[1][i] = self._K_LP(self.subseq_length, m, Y[i], Y[i])

        # Calculating the kernel value for the documents
        for i in range(len_X):
            for j in range(len_Y):
                if gram[i][j] == 0:
                    if X != Y:
                        resultKernel = self._gram_matrix_element(X[i], Y[j], m, normalizing_kernel[0][i],
                                                                 normalizing_kernel[1][j])
                    else:
                        resultKernel = self._gram_matrix_element(X[i], Y[j], m, normalizing_kernel[0][i],
                                                                 normalizing_kernel[0][j])
                    gram[i][j] = resultKernel
                    # Exploiting the symmetric property of the gram matrix
                    if (j < len_X) and (i < len_Y) and (i != j):
                        gram[j][i] = resultKernel
        return gram

    def _gram_matrix_element(self, s, t,m, sdkvalue1, sdkvalue2):
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
                return self._K_LP(self.subseq_length,m, s, t) / \
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
            f1 = f1_score(test_labels, prediction, average=None)
            recall = recall_score(test_labels, prediction, average=None)
            precision = precision_score(test_labels, prediction, average=None)
            print("accuracy", correct_predictions / len(test_labels))
            print("F1", f1)
            print("Recall", recall)
            print("precision", precision)

train_input = "train_20_iter_0.csv"
test_input = "test_8_iter_0.csv"
train_data, train_label, test_data, test_label = ds.reduce_data(train_input, test_input)

SVM_SSK = SVM("SVM_SSK", 3)

t_start = time()
SVM_SSK.train(train_data[:5], train_label[:5])
print('Model trained in %.3f seconds' % (time() - t_start))
gc.collect()
t_start = time()
predicted_output = SVM_SSK.predict(train_data[:5], test_data[:2])
print('Model prediction time %.3f seconds' % (time() - t_start))
SVM_SSK.evaluatePrediction(predicted_output, test_label[:2])