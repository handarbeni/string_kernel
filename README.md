# Deep Image Colorization

Implementation of string kernel, word kernel and n-gram kernel for Support Vector Machine model for text classification.  This project was developed as a part of **DD2434-Machine Learning, Advanced Course** course at **KTH Royal Institute of Technology** Stockholm. 

### Authors
[Muhammad Irfan Handarbeni], Polyxeni Parthena Ioannidou, [Fadhil Mochammad], Deepika Anantha Padmanaban

The implementions are in Python using NLTK and Skicit-Learn Library.

### Abstract
The purpose of this project is to read, understand and re-implement the kernel methods in text classification. The paper that we followed is "Text Classification using String Kernels" by [Lodhi, Saunders, Shawe-Taylor, Cristianini and Watkins]. In this project we focus on classifying text documents using kernels and support vector machines. We use kernels to extract features from data unambiguously. More specifically, Kernel functions return the inner product between the mapped data points in a higher dimensional space, where the data could become more easily separated or better structured.

### Overview
We consider three different types of kernels as mentioned in the original paper, namely, String Subsequence
Kernel(SSK), n-Grams Kernel(NGK) and Word-Kernel(WK). We tried re-implementing each of the kernels
and used them for training our SVMs and later used the model for prediction. We also observe the effect
of varying the hyper-parameters ’n’ and ’λ’ for the SSK and ’n’ for NGK. We compare the results obtained
from our experiments with the original results in the paper and discuss the main implications of our results.
We also, try providing an extension to the framework of SSK, in order reduce the computation time, called
Lambda Pruning. Finally, we conclude with the mention of a few state-of-the-art method in text classification
and their advantages over these traditional methods.

[//]: # 

   [Muhammad Irfan Handarbeni]: <https://github.com/handarbeni>
   [Fadhil Mochammad]: <https://github.com/fadhilmch>
   [Lodhi, Saunders, Shawe-Taylor, Cristianini and Watkins]: <http://www.jmlr.org/papers/volume2/lodhi02a/lodhi02a.pdf>