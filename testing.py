#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:58:50 2019
SENSITIVITY CLASSIFICATION - Training
@author: Raj Kumar Shrivastava
"""
import pandas as pd
import numpy as np
import pickle
import time
import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import tensorflow as tf
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 

class NeuralNet():
    def __init__(self):                           #initializing hyperparameters    
        self.window_size = 7        #window_size should be same as that in training
        self.word_index = pickle.load(open('word_index.txt','rb'))
        self.word_voc_size = len(self.word_index) + 1              # +1 for padding of the input vector in case where new words are encountered.
        self.mini_batch_size_classifier = 512

    def create_corpus(self, testing_data_file):            #data cleansing
        
        print("Reading testing data...")
        data=pd.read_csv(testing_data_file)   #each row should contain: [label, mail]
       
        mails    = data.iloc[:, 1]   
        polarity = data.iloc[:, 0]   
        
        print("Length of testing dataset: ", len(mails))
        mails=list(mails)
        
        porter = PorterStemmer()   #Stemming to root words 

        filtered_mails=[]
        pols=[]
        for i, mail in enumerate(mails):     #each mail
            if(type(mail)==float):
                continue
            
            mail = mail.lower()
            mail = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",mail).split()) #removing punctuations
            mail = re.sub(r'(.)\1+', r'\1\1', mail)     #remove repeating characters (for eg., yessss -> yes)

            stemmed_words = [porter.stem(word) for word in word_tokenize(mail)]     #reducing words to their root words
            
            filtered_words = []
            for word in stemmed_words:
                if word in self.word_index:
                    filtered_words.append(word)
                
            if len(filtered_words)>=2 and len(filtered_words)<=300:
                filtered_mails.append(filtered_words)
                pols.append(polarity.iloc[i])

        return filtered_mails, pols
    
    def classifier_dataset(self, filtered_mails, polarity):
        x = []
        y = []
        
        for mail_no, mail in enumerate(filtered_mails):  #cycling through each mail
            mail_len = len(mail)
            for i, word in enumerate(mail):     #cycling through each word
                if word in self.word_index:
                    current=self.word_index[word]        #index of current_word
                else:
                    continue
        
                contexts_a=[]               #indexes of before words
                contexts_b=[]               #indexes of after words
                for j in range(i-self.window_size, i):
                    if j>=0 and mail[j] in self.word_index:
                        contexts_a.append(self.word_index[mail[j]])
                    else:
                        contexts_a.append(self.word_voc_size)
                    
                for j in range(i+1, i+1+self.window_size):
                    if j<mail_len and mail[j] in self.word_index:
                        contexts_b.append(self.word_index[mail[j]])    
                    else:
                        contexts_b.append(self.word_voc_size)
                
                inp = contexts_a + [current] + contexts_b
                
                if(polarity[mail_no]==1):     #1-sensitive
                    pol=[1,0]     
                else:
                    pol=[0,1]      #0-non-sensitive
                x.append(inp)
                y.append(pol)

        return x, y  
    
    def test_classifier(self, test_x, test_y):
        def accuracy(predictions, labels):
            return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                    / predictions.shape[0])
        
        length = len(test_x)
        print("No of input data: ", length)        
        
        self.a = mini_batches_x = [ test_x[k:k+self.mini_batch_size_classifier] for k in range(0, length, self.mini_batch_size_classifier) ]
        mini_batches_y = [ test_y[k:k+self.mini_batch_size_classifier] for k in range(0, length, self.mini_batch_size_classifier) ]
        
        with tf.Session() as sess:
            #Reloading the trained model
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph('./saved_model/my-model-2.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./saved_model'))
            graph = tf.get_default_graph()
            X = graph.get_tensor_by_name("X:0")            
            output_layer = graph.get_tensor_by_name('output_layer_s:0')
            print("Model restored.")
            
            predicted = []
            actual = []
            for mini_count in range(len(mini_batches_x)):
                output = sess.run(output_layer, feed_dict={X:mini_batches_x[mini_count]})
                
                predicted_temp = np.argmax(output, 1)
                predicted.extend(list(predicted_temp))  
                
                actual_temp = np.argmax(mini_batches_y[mini_count], 1)
                actual.extend(list(actual_temp))
            
            results = confusion_matrix(actual, predicted) 
            print('Confusion Matrix :')
            print(results) 
            print('Accuracy Score :',accuracy_score(actual, predicted) )
            
        
#DRIVER CODE
if __name__=='__main__':
    np.random.seed(0)
    model = NeuralNet()
    
    #Feeding the name (with path, if in a different folder) of the testing data file in csv format.
    #Each row is in the following format : [sender email id, receiver email ids, subject, message]
    testing_data_file = 'testing_data.csv'    

    print("Generating corpus...", time.ctime(time.time()))    
    filtered_mails, polarity= model.create_corpus(testing_data_file)
    
    print("Generating classifier testing_data...", time.ctime(time.time()))
    x, y = model.classifier_dataset(filtered_mails, polarity)   #1 = 196292     #0 = 321109 

    print("Testing classifier...", time.ctime(time.time()))
    model.test_classifier(x, y)      #training classifier neural network
        
    print("PROCESS COMPLETED.")