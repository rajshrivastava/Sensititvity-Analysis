#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:58:50 2019
Skipgram generator
@author: Raj Kumar Shrivastava
"""
import pandas as pd
import numpy as np
import pickle
import time
import random
import re
import math
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter 
import tensorflow as tf

class NeuralNet():
    def __init__(self, config):                                     #initializing hyperparameters
        self.n = config['n']        #length of skipgram vectors        
        self.window_size = config['window_size']     #neighbourhood length of each side for a word
        self.top_n_words = config['top_n_words']     #top_n most frequent words to be considered
        self.mini_batch_size_skipgram=config['mini_batch_size_skipgram']
        self.skipgram_epochs = config['skipgram_epochs']
        self.skipgram_eta = config['skipgram_eta']
    
    def create_corpus(self, datafile_name):            #data cleansing       
        print("Reading corpus...")
        data=pd.read_csv(datafile_name)   #each row should contain: [label, mail]   
        mails    = data.iloc[:, 1]   #full
        print("Length of dataset: ", len(mails))
        
        mails=list(mails)
        porter = PorterStemmer()   #Stemming to root words 
        stop_words=['k','m','t','d','e','f','g','h','i','u','r','I','im',\
                    'ourselves', 'hers', 'between', 'yourself', 'again', \
                    'there', 'about', 'once', 'during', 'out', 'very', \
                    'having', 'with', 'they', 'own', 'an', 'be', 'some', \
                    'for', 'do', 'its', 'yours', 'such', 'into', 'of', \
                    'most', 'itself', 'other', 'off', 'is', 's', 'am', \
                    'or', 'who', 'as', 'from', 'him', 'each', 'the', \
                    'themselves', 'until', 'below', 'are', 'we', 'these', \
                    'your', 'his', 'through', 'don', 'nor', 'me', 'were', \
                    'her', 'more', 'himself', 'this', 'should', 'our', \
                    'their', 'while', 'above', 'both', 'to', 'ours', 'had', \
                    'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',\
                    'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',\
                    'yourselves', 'then', 'that', 'because', 'what', 'over', \
                    'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you',\
                    'herself', 'has', 'just', 'where', 'too', 'only', 'myself',\
                    'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',\
                    'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',\
                    'how', 'further', 'was', 'here', 'than'];
        
        word_counts = {}   #dictionary to store all words and their counts

        filtered_mails=[]
        for i, mail in enumerate(mails):     #each mail
            if(type(mail)==float):
                continue
            mail = mail.lower()
            mail = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",mail).split()) #remove puncs
            mail = re.sub(r'(.)\1+', r'\1\1', mail)     #remove repeating characterssss

            stemmed_words = [porter.stem(word) for word in word_tokenize(mail)]
            
            filtered_words=[]            
            for word in stemmed_words:
                if (word not in stop_words) and word.isdigit() == False:
                    filtered_words.append(word)
                    if word in word_counts:
                        word_counts[word] += 1    #for existing words
                    else:
                        word_counts[word] = 1     #for new words
                
            if len(filtered_words)>=2:
                filtered_mails.append(filtered_words)
        
        #Assuming that words which have occured rarely are insignificant or noisy words,
        #we eliminate the less frequent words
        k = Counter(word_counts)
        high = k.most_common(self.top_n_words)
        word_counts = dict(high)
        
        self.word_voc_size = len(word_counts)
        print("Word vocabulary size = ",self.word_voc_size)
        
        self.word_index = dict((word, i) for i, word in enumerate(word_counts))     #word  -> index
        self.index_word = dict((i, word) for i, word in enumerate(word_counts))     #index -> word
        
        #Saving all the dictionaries to the disk.
        #Will be used for training and testing and creating their respective datasets.
        out_file=open('word_index.txt','wb')
        pickle.dump(self.word_index,out_file)
        out_file.close()    
        
        out_file=open('index_word.txt','wb')
        pickle.dump(self.index_word,out_file)
        out_file.close()
        
        return filtered_mails
        
    def skipgram_dataset(self, filtered_mails):
        targets = []
        contexts= []
        for sentence in filtered_mails:
            sent_len = len(sentence)
            
            for i, word in enumerate(sentence):                
                if word not in self.word_index: continue
            
                #A random window size between 1 and specified (maximum) window_size will be used.
                #This ensures that more weightage is given to the nearer word than the farther words
                random_window_size = random.randint(1,self.window_size)  
                lower = max(i-random_window_size, 0)
                upper = min(sent_len-1, i+random_window_size)
                
                for j in range(lower, i):                               #left window words
                    if sentence[j] in self.word_index:
                        targets.append(self.word_index[word])
                        contexts.append(self.word_index[sentence[j]])
                        
                for j in range(i+1, upper+1):                           #right window words
                    if sentence[j] in self.word_index:
                        targets.append(self.word_index[word])
                        contexts.append(self.word_index[sentence[j]])
    
        return targets, contexts
    
    def train_skipgram(self, targets, contexts):
        length = len(targets)
        print("Length of training data: ", length)        
        print("Minibatch size: ", self.mini_batch_size_skipgram)
        
        #graph architecture
        graph=tf.Graph()
        with graph.as_default():
            tf.set_random_seed(1)
            self.w1   = tf.Variable(tf.random_uniform([self.word_voc_size+1, self.n], -1.0, 1.0),name='w1')
    
            self.nce_w2   = tf.Variable(tf.truncated_normal([self.word_voc_size, self.n], stddev=1.0/math.sqrt(self.n)), name='nce_w2')
            self.nce_b2   = tf.Variable(tf.zeros([self.word_voc_size]),name='nce_b2')
            
            X = tf.placeholder(tf.int32, shape = [None])
            Y = tf.placeholder(tf.int32, shape = [None, 1])  
            
            h = tf.nn.embedding_lookup(self.w1, X) 
            print("h: ",h.get_shape())
            
            loss = tf.reduce_mean(tf.nn.nce_loss(weights = self.nce_w2,
                                                 biases = self.nce_b2,
                                                 labels = Y,    #[batch_size, num_true]
                                                 inputs = h,    #[batch_size, dim]
                                                 num_sampled = 64,
                                                 num_classes = self.word_voc_size))
            
            optimizer = tf.train.AdamOptimizer(self.skipgram_eta).minimize(loss)
            #optimizer = tf.train.GradientDescentOptimizer(self.eta).minimize(loss)
            
        #graph architecture end####
              
        # train on mini batches
        print("Training started at ", time.ctime(time.time()) )
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            for epo in range(self.skipgram_epochs):
                #Re-shuffling the training data for each epoch to improve generalization
                temp = list(zip(targets, contexts))
                random.shuffle(temp) 
                targets, contexts = zip(*temp)
                
                mini_batches_x = [ targets[k:k+self.mini_batch_size_skipgram] for k in range(0, length, self.mini_batch_size_skipgram)]
                mini_batches_y = [ contexts[k:k+self.mini_batch_size_skipgram] for k in range(0, length, self.mini_batch_size_skipgram)]
                loss_sum = 0
                for mini_count in (range(len(mini_batches_x))):
                    batch_x = mini_batches_x[mini_count]
                    batch_y = mini_batches_y[mini_count]  
                    batch_y = np.array(batch_y).reshape(len(batch_y),1)
                    _, mini_loss = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
                    
                    loss_sum += mini_loss
                print("Iteration", epo+1, "\t| Loss =", loss_sum/len(mini_batches_x), "\t| Time =",time.ctime(time.time()))
            print("Skipgram training completed at ", time.ctime(time.time()) )
            
            np.save('skipgrams.npy', self.w1.eval())
         
#DRIVER CODE
def skipgram_generator(datafile_name):
    np.random.seed(0)
    
    #hyperparameters configuration for training skipgrams
    config={'n':50, 'window_size':10, 'top_n_words': 20000, 'mini_batch_size_skipgram':256, 'skipgram_epochs':3, \
            'skipgram_eta':0.0003}
     
    model = NeuralNet(config)
    
    print("Generating corpus...", time.ctime(time.time()))    
    filtered_mails = model.create_corpus(datafile_name)
    
    print("Generating skipgram training_data...", time.ctime(time.time()))
    targets, contexts = model.skipgram_dataset(filtered_mails)
        
    print("Training skipgrams...", time.ctime(time.time()))
    model.train_skipgram(targets, contexts)      #training skipgram neural network