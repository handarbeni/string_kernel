import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import LinearSVC, SVC

def generate_dataset(df_train, df_test, n_train, n_test, exp_iter):
    train = df_train[df_train.label == 0].sample(n=n_train).append(df_train[df_train.label == 1].sample(n=n_train).append(df_train[df_train.label == 2].sample(n=n_train).append(df_train[df_train.label == 3].sample(n=n_train)))).sample(frac=1).reset_index(drop=True)
    test = df_test[df_test.label == 0].sample(n=n_test).append(df_test[df_test.label == 1].sample(n=n_test).append(df_test[df_test.label == 2].sample(n=n_test).append(df_test[df_test.label == 3].sample(n=n_test)))).sample(frac=1).reset_index(drop=True)
    
    train.to_csv('../final_dataset/train_'+str(n_train*4)+'_iter_'+str(exp_iter)+'.csv', sep=',', index=False)
    test.to_csv('../final_dataset/test_'+str(n_test*4)+'_iter_'+str(exp_iter)+'.csv', sep=',', index=False)

def load_dataset(n_train, n_test, exp_iter):
    train = pd.read_csv('../final_dataset/train_'+str(n_train*4)+'_iter_'+str(exp_iter)+'.csv')
    test = pd.read_csv('../final_dataset/test_'+str(n_test*4)+'_iter_'+str(exp_iter)+'.csv')
    
    train_docs = train.body
    test_docs = test.body
    train_labels = train.label.values
    test_labels = test.label.values
    
    return train_docs, test_docs, train_labels, test_labels

def ng_kernel(doc_1,doc_2,n):
    tfidfVectorizer = TfidfVectorizer(analyzer = "char",tokenizer = None,preprocessor = None,ngram_range=(n, n))
    docs = tfidfVectorizer.fit_transform([doc_1,doc_2])
    docs = docs.toarray()
    
    score = np.dot(docs[0], docs[1])/np.sqrt(np.dot(docs[0], docs[0])*np.dot(docs[1], docs[1]))
    
    return score

def run_experiment(n_gram, n_train, n_test):
    f1s = [[],[],[],[]]
    precisions = [[],[],[],[]]
    recalls = [[],[],[],[]]

    labs = [[0], [1], [2], [3]]
    labels = ['acq', 'corn', 'crude', 'earn']
    
    result = []
    for exp_iter in range(10):
        train_docs, test_docs, train_labels, test_labels = load_dataset(n_train, n_test, exp_iter)
        train_gram_matrix = np.zeros((n_train*4,n_train*4))
        test_gram_matrix = np.zeros((n_test*4,n_train*4))
        for i in range(n_train*4):
            for j in range(i+1,n_train*4):
                train_gram_matrix[i][j] = ng_kernel(train_docs[i],train_docs[j], n_gram)

        train_gram_matrix = train_gram_matrix + train_gram_matrix.T + np.eye(n_train*4)

        for i in range(n_test*4):
            for j in range(n_train*4):
                test_gram_matrix[i][j] = ng_kernel(test_docs[i],train_docs[j], n_gram)

        model = SVC(kernel='precomputed')
        model.fit(train_gram_matrix, train_labels)

        predictions = model.predict(test_gram_matrix)
    
    #print('===============================experiment_'+str(exp_iter)+'===================================')
        for lab in labs:
            f1 = f1_score(test_labels, predictions,lab,average='macro')
            precision = precision_score(test_labels, predictions,lab,average='macro')
            recall = recall_score(test_labels, predictions,lab, average='macro')

            f1s[lab[0]].append(f1)
            precisions[lab[0]].append(precision)
            recalls[lab[0]].append(recall)
        
        #print(labels[lab[0]] + ' # f1-score = ' + str(f1) + ", precision = " + str(precision) + ", recal = " + str(recall))

        
    for lab in range(4):
        #print('================================'+labels[lab]+'=================================')
        avg_f1, avg_precision, avg_recall = np.mean(f1s[lab]), np.mean(precisions[lab]), np.mean(recalls[lab])
        sd_f1, sd_precision, sd_recall = np.std(f1s[lab]), np.std(precisions[lab]), np.std(recalls[lab])
        
        result.append([n_gram, labels[lab], avg_f1, sd_f1, avg_precision, sd_precision, avg_recall, sd_recall])

        #print("mean f1 = "+str(avg_f1)+", precision = "+str(avg_precision)+", recall = "+str(avg_recall))
        #print("standard deviation f1 = "+str(sd_f1)+", precision = "+str(sd_precision)+", recall = "+str(sd_recall))
    return result

if __name__ == "__main__":
	category = ['earn', 'acq', 'crude', 'corn']
	mapping = {'acq': 0, 'corn': 1, 'crude': 2, 'earn' : 3}
	df_train = pd.read_csv('../final_dataset/train_multiclass.csv')
	df_train = df_train[df_train.label.isin(category)]
	df_train = df_train.replace({'label':mapping})
	df_train.reset_index(inplace=True, drop=True)
	df_train.head()

	df_test = pd.read_csv('../final_dataset/test_multiclass.csv')
	df_test = df_test[df_test.label.isin(category)]
	df_test = df_test.replace({'label':mapping})
	df_test.reset_index(inplace=True, drop=True)
	df_test.head()

	n_train = 25
	n_test = 5
	
	for i in range(10):
	    generate_dataset(df_train, df_test, n_train, n_test, i)

	n_gram = [3,4,5,6,7,8,10,12,14]

	result = []

	for ng in n_gram:
	    temp_res = run_experiment(ng,n_train,n_test)
	    result += temp_res

	result = pd.DataFrame(result, columns=['n', 'category', 'mean_f1', 'sd_f1', 'mean_precision', 'sd_precision', 'mean_recall', 'sd_recall'])
	result.to_csv('../result/100_train_20_test_result.csv', sep=',', index=False)