from bs4 import BeautifulSoup
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np


labels = ["earn", "acq", "crude", "corn"]

def iterateModels(numOfIterations, models, predictionsArray, testLabels):

    F1 = [[0.0 for x in range(0,len(labels))] for y in range(0,len(models))]
    precision = [[0.0 for x in range(0,len(labels))] for y in range(0,len(models))]
    recall = [[0.0 for x in range(0,len(labels))] for y in range(0,len(models))]

    for i in range(numOfIterations):

        for j in range(len(models)):

            F1_temp = f1_score(testLabels, predictionsArray[j], average=None)
            recall_temp = recall_score(testLabels, predictionsArray[j], average=None)
            precision_temp = precision_score(testLabels, predictionsArray[j], average=None)

            for k in range(len(labels)):
                F1[j][k] = F1_temp[k]
                recall[j][k] = recall_temp[k]
                precision[j][k] = precision_temp[k]

    F1 = compute_F1_std_mean(F1, len(models), len(labels))
    precision = compute_precision_std_mean(precision, len(models), len(labels))
    recall = compute_recall_std_mean(recall, len(models), len(labels))

    for i in range(len(labels)):
        print(labels[i])
        for k in range(len(models)):
            print("\t\t" + models[k].name)
            print("\t\t\t" + str(F1[k][i][0]) + "\t" + str(F1[k][i][1]) + "\t" + str(precision[k][i][0]) + "\t" + str(
                precision[k][i][1]) + "\t" + str(recall[k][i][0]) + "\t" + str(recall[k][i][1]))


# N is len(labels), M is len(models)
def compute_F1_std_mean(F1, M, N):

    for j in range(M):
        for k in range(N):
            F1_mean = np.mean(F1[j][k])
            F1_std = np.std(F1[j][k])
            F1[j][k] = [F1_mean, F1_std]

    return F1


def compute_precision_std_mean(precision, M, N):

    for j in range(M):
        for k in range(N):
            precision_mean = np.mean(precision[j][k])
            precision_std = np.std(precision[j][k])
            precision[j][k] = [precision_mean, precision_std]

    return precision


def compute_recall_std_mean(recall, M, N):

    for j in range(M):
        for k in range(N):
            recall_mean = np.mean(recall[j][k])
            recall_std = np.std(recall[j][k])
            recall[j][k] = [recall_mean, recall_std]

    return recall
