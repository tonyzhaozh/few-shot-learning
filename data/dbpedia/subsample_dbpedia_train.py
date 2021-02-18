import csv
import pandas as pd
import random
train_data = pd.read_csv('data/dbpedia/train.csv')
train_sentences = train_data['Text']
train_titles = train_data['Title']
train_labels = train_data['Class']
c = list(zip(train_sentences, train_titles, train_labels))
random.shuffle(c)
train_sentences, train_titles, train_labels = zip(*c)
train_sentences = train_sentences[0:50000]
train_titles = train_titles[0:50000]
train_labels = train_labels[0:50000]    
with open('data/dbpedia/train_subset.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')        
    for i, line in enumerate(train_sentences):
        if i == 0:
            f.write('Class,Title,Text\n')
        else:
            writer.writerow([str(train_labels[i]),str(train_titles[i]),str(train_sentences[i])])