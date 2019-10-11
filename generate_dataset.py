import numpy as np
import pandas as pd
import os
import bs4
import string
import re

def generate_dataset(file_name, datatype):
    reader = open(data_path + file_name, 'r')
    file = reader.read()
    
    parser = bs4.BeautifulSoup(file, 'html.parser')
    articles = parser.find_all('text')
    
    categories = ['earn', 'acq', 'crude', 'corn']
    
    bodies = []
    labels = [[] for _ in range(4)]
    for article in articles:
            
        body = article.find('body').string
        body = body.strip()
        body = body.replace(' \x03', '')
        re.sub("\s\s+", " ", body)
        bodies.append(body)
        
        i = 0
        for cat in categories:
            topics = article.find('topics')
            flag = False
            for topic in topics.find_all('d'):
                if topic.string == cat:
                    labels[i].append(1)
                    flag = True
                    break
            if flag == False:
                labels[i].append(0)
            i = i+1
    
    df_dataset = pd.concat([pd.DataFrame({'body': bodies}), pd.DataFrame(labels).T], axis=1)
    df_dataset.columns = ['body', 'earn', 'acq', 'crude', 'corn']
    df_dataset.to_csv(datatype+'_single.csv', sep=',', index=False)

def generate_dataset_multiclass(file_name, datatype):
    reader = open(data_path + file_name, 'r')
    file = reader.read()
    
    parser = bs4.BeautifulSoup(file, 'html.parser')
    articles = parser.find_all('text')
    
    categories = ['earn', 'acq', 'crude', 'corn']
    
    bodies = []
    labels = []
    for article in articles:
            
        body = article.find('body').string
        body = body.strip()
        body = body.replace(' \x03', '')
        re.sub("\s\s+", " ", body)
        bodies.append(body)
       
        flag = False
        for cat in categories:
            topics = article.find('topics')
            
            for topic in topics.find_all('d'):
                if topic.string == cat:
                    labels.append(topic.string)
                    flag = True
                    break
            if flag == True:
                break
        if flag == False:
            labels.append(topics.find_all('d')[0].string)
    
    print(len(bodies))
    print(len(labels))
    df_dataset = pd.DataFrame({'body': bodies, 'label':labels})
    df_dataset.to_csv(datatype+'_multiclass.csv', sep=',', index=False)

if __name__ == '__main__':
	data_path = '../preprocessed_dataset/'
	file_names = os.listdir(data_path)

	generate_dataset('train_stemmed.sgm', 'train')
	generate_dataset('test_stemmed.sgm', 'test')

	generate_dataset_multiclass('train_stemmed.sgm', 'train')
	generate_dataset_multiclass('test_stemmed.sgm', 'test')