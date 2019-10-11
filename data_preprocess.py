import nltk
import os
import bs4
import string

def remove_stopwords_punc(text):
    text = ' '.join([word for word in text.split() if word not in stopwords])
    text = text.translate(str.maketrans(punctuation, ' '*len(punctuation)))
    #text = ''.join([word for word in list(text) if word not in punctuation])
    return text

 def stem_words(text,stemmer):
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def write_file(docs, labels, ids_doc, filename):
    f = open(filename, 'w')
    
    for doc in docs:
        label = labels[doc]
        id_doc = ids_doc[doc]
        
        f.write('<TEXT ID=' + str(id_doc) + '>\n')
        f.write('<TOPICS>\n')
        for lab in label:
            f.write('<D>'+lab+'</D>')
        f.write('</TOPICS>\n')
        f.write('<BODY>\n')
        f.write(doc)
        f.write('</BODY>\n')
        f.write('</TEXT>\n')
        
    f.close

if __name__ == '__main__':

	nltk.download('stopwords')
	data_path = '../dataset/'
	file_names = os.listdir(data_path)

	files = []
	for name in file_names:
	    reader = open(data_path + name, 'r')
	    file = reader.read()
	    files.append(file)

	stopwords = nltk.corpus.stopwords.words('english')
	punctuation = string.punctuation

	labels = {}
	ids_doc = {}
	train = []
	test = []

	stemmer = nltk.stem.snowball.SnowballStemmer("english")

	id_doc = 0
	for file in files:
	    parser = bs4.BeautifulSoup(file, 'html.parser')
	    articles = parser.find_all('reuters')
	    
	    for article in articles:
	        body = article.find('body')
	        if body != None:
	            body_text = body.string
	            data_type = article.get('lewissplit')
	            topics_list = article.find('topics')
	            
	            topics = []
	            for topic in topics_list.find_all('d'):
	                topics.append(topic.string)
	            
	            if len(topics_list)>0:
	                body_text = remove_stopwords_punc(body_text)
	                body_text = stem_words(body_text, stemmer)
	                if data_type == 'TRAIN':
	                    train.append(body_text)
	                elif data_type == 'TEST':
	                    test.append(body_text)
	            else:
	                continue
	            
	            labels[body_text] = topics
	            ids_doc[body_text] = id_doc
	            id_doc += 1

	write_file(train, labels, ids_doc, '../preprocessed_dataset/train_stemmed.sgm')
	write_file(test, labels, ids_doc, '../preprocessed_dataset/test_stemmed.sgm')