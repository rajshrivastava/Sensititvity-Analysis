# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:56:10 2018
SENTIMENT CLASSIFICATION USING PREDICTION MODEL
@author: Raj
"""
import numpy as np
import pickle
import re
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import tensorflow as tf

class NeuralNet():
    def __init__(self):                           #initializing hyperparameters     
        self.window_size = 7        #window_size should be same as that in training
        self.word_index = pickle.load(open('word_index.txt','rb'))
        self.word_voc_size = len(self.word_index) + 1              # +1 for padding of the input vector in case where new words are encountered.
        
    def create_corpus(self, mails):            #data cleansing
        mails=list(mails)
        
        porter = PorterStemmer()   #Stemming to root words 

        filtered_mails=[]
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

        return filtered_mails
    
    def format_data(self, filtered_mails):
        x = []
        
        for mail in filtered_mails:  #cycling through each mail
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
                
                x.append(inp)

        return x  
    
    def predict(self, data_x):
        with tf.Session() as sess:
            #Reloading the trained model
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph('./saved_model/my-model-2.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./saved_model'))
            graph = tf.get_default_graph()
            X = graph.get_tensor_by_name("X:0")            
            output_layer = graph.get_tensor_by_name('output_layer_s:0')
            print("Model restored.")
            
            self.a = outputs = sess.run(output_layer, feed_dict={X:data_x})
                   
        pos_score = 0
        neg_score = 0
        pos_count = 0
        neg_count = 0
        
        for pred in outputs:
            pos_score += pred[0]
            neg_score += pred[1]
            if(pred[0] > pred[1]):
                pos_count += 1
            else:
                neg_count += 1
       
        print('')
        if(pos_count>0):
            prediction="SENSITIVE EMAIL"
        else:
            prediction="NON-SENSITIVE EMAIL"
        print(prediction)    
        
        
#DRIVER CODE
if __name__=='__main__':
    np.random.seed(0)

    model = NeuralNet()
    
    data = input("Please provide an input:\n")
        
    filtered_mail = model.create_corpus(sent_tokenize(data))
   
    x = model.format_data(filtered_mail)    
    model.predict(x)      #neural network