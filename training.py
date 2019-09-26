#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:58:50 2019
@author: Raj Kumar Shrivastava

SENSITIVITY CLASSIFICATION - Training
"""
import pandas as pd
import numpy as np
import pickle
from skipgram_generator import skipgram_generator
import time
import random
import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import tensorflow as tf

class NeuralNet():
    def __init__(self, config):                           #initializing hyperparameters        
        self.m = config['m']    #hidden layer size
        self.classifier_epochs = config['classifier_epochs']
        self.window_size = config['window_size']
        self.mini_batch_size_classifier=config['mini_batch_size_classifier']
        self.k = config['k']                        #negative samples size for noise contrastive estimation
        self.alpha_pred = config['alpha_pred']      #for the weighted average of the two losses in this hybrid model
        
        #Loading files from disk
        self.word_index = pickle.load(open('word_index.txt','rb'))
        self.skipgrams = np.load('skipgrams.npy')
        
        self.n = len(self.skipgrams[0])    #length of skipgram vector+6
        self.word_voc_size = len(self.skipgrams)
            
    def create_corpus(self, training_data):            #data cleansing
        print("Reading training data...")
        data=pd.read_csv(training_data)
       
        polarity = data.iloc[:, 0]
        mails    = data.iloc[:, 1]
        
        print("Length of training dataset: ", len(mails))
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
        x_noise = []
        y = []
        
        for mail_no, mail in enumerate(filtered_mails):  #cycling through each mail
            mail_len = len(mail)
            for i, word in enumerate(mail):     #cycling through each word
                if word in self.word_index:
                    current=self.word_index[word]        #index of current_word
                    noise=current
                else:
                    continue
                
                while(noise==current):
                    noise=random.randint(0, self.word_voc_size-1)
        
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
                noise_inp = contexts_a + [noise] + contexts_b
                
                if(polarity[mail_no]==1):     #1-sensitive
                    pol=[1,0]     
                else:
                    pol=[0,1]      #0-non-sensitive
                x.append(inp)
                x_noise.append(noise_inp)
                y.append(pol)

        return x, y, x_noise    
    
    def train_classifier(self, dataset_x, dataset_y, dataset_x_noise):
        
        #Shuffling the data before splitting into training and validaton set
        temp = list(zip(dataset_x, dataset_y, dataset_x_noise))      
        random.shuffle(temp) 
        dataset_x, dataset_y, dataset_x_noise = zip(*temp)
        
        length = len(dataset_x)
        print('->Length of training dataset: ', length)
        
        #Assigning 75% of the data for training, 25% for testing.
        #This ratio can vary depending on the availability of the total data
        idx1 = int(0.75*length)
        train_x = dataset_x[:idx1]
        train_y = dataset_y[:idx1]
        train_x_noise = dataset_x_noise[:idx1]
        
        train_len = len(train_x)
        
        valid_x = dataset_x[idx1:]
        valid_y = dataset_y[idx1:]
        
        valid_len = len(valid_x)
    
        print("Length of training data: ", train_len) 
        print("Length of validation data: ", valid_len) 
        
        print("Minibatch size: ", self.mini_batch_size_classifier)
        print("->Word vocabulary size: ", self.word_voc_size)
        
        #Now, we create the architecture of the neural network and define the operations to be performed
        graph=tf.Graph()
        with graph.as_default():
            tf.set_random_seed(1)
            
            input_len = self.window_size*2 + 1
            print("Input neurons: ", input_len)
            X = tf.placeholder(tf.int32, [None, input_len], name='X')
            X_noise = tf.placeholder(tf.int32, [None, input_len])
            Y = tf.placeholder(tf.float32, [None, 2])
            Y_c = tf.placeholder(tf.int32, [None,1])
            #keep_prob = tf.placeholder(tf.float32)
            
            #self.w1   = tf.Variable(tf.truncated_normal([self.word_voc_size+1, self.n], mean=0, stddev=0.1), name='w1') #embedding matrix
            self.w1 = tf.Variable(tf.convert_to_tensor(self.skipgrams, np.float32), name='w1')
            print("Shape of self.w1", self.w1.get_shape())
            self.w2   = tf.Variable(tf.truncated_normal([input_len*self.n, self.m], mean=0, stddev=0.1), name='w2') #lookup->linear1
            self.b2   = tf.Variable(tf.truncated_normal([self.m], mean=0, stddev=0.1), name='b2')
            self.w3   = tf.Variable(tf.truncated_normal([self.m, self.m], mean=0, stddev=0.1), name='w3')       #linear1->htanh
            self.b3   = tf.Variable(tf.truncated_normal([self.m], mean=0, stddev=0.1), name='b3')
            self.w4_s = tf.Variable(tf.truncated_normal([self.m, 2], mean=0, stddev=0.1), name='w4')    #htanh->linear2_s
            self.b4_s = tf.Variable(tf.truncated_normal([2], mean=0, stddev=0.1), name='b4')
            self.w5   = tf.Variable(tf.truncated_normal([2, 2], mean=0, stddev=0.1), name='w5')     #linear2_s->softmax
            self.b5   = tf.Variable(tf.truncated_normal([2], mean=0, stddev=0.1), name='b5')
            
            self.w4_c = tf.Variable(tf.truncated_normal([self.m, 1], mean=0, stddev=0.1), name='w4_c')    #htanh->linear2_c
            self.b4_c = tf.Variable(tf.truncated_normal([1], mean=0, stddev=0.1), name='b4_c')
    
            look_ups = tf.nn.embedding_lookup(self.w1, X)               
            lookup_layer=tf.reshape(look_ups, shape=(-1, input_len*self.n) )        
            linear_layer1= tf.add(tf.matmul(lookup_layer, self.w2), self.b2, name='linear_layer1')
            htanh_layer  = tf.tanh(tf.matmul(linear_layer1, self.w3) + self.b3, name='htanh_layer')
            linear_layer2_s = tf.add(tf.matmul(htanh_layer, self.w4_s), self.b4_s, name='linear_layer2_s')
            linear_layer2_c = tf.add(tf.matmul(htanh_layer, self.w4_c), self.b4_c, name='linear_layer2_c')
            output_layer_s = tf.add(tf.matmul(linear_layer2_s, self.w5), self.b5, name='output_layer_s') #softmax loss
            cross_entropy_s = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output_layer_s))
            
            
            look_ups_noise = tf.nn.embedding_lookup(self.w1, X_noise)
            lookup_layer_noise=tf.reshape(look_ups_noise, shape=(-1, input_len*self.n) )
            linear_layer1_noise = tf.add(tf.matmul(lookup_layer_noise, self.w2), self.b2, name='linear_layer1_noise')
            htanh_layer_noise  = tf.tanh(tf.matmul(linear_layer1_noise, self.w3) + self.b3, name='htanh_layer_noise')
            linear_layer2_c_noise = tf.add(tf.matmul(htanh_layer_noise, self.w4_c), self.b4_c, name='linear_layer2_noise')
                         
            context_score_nce = tf.divide( tf.exp(linear_layer2_c),\
                                          tf.add( tf.exp(linear_layer2_c),\
                                                 tf.scalar_mul(self.k, tf.exp(linear_layer2_c_noise)) )  )
            
            cross_entropy_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_c, logits=context_score_nce))
                    
            
            cross_entropy = tf.add(tf.scalar_mul(self.alpha_pred, cross_entropy_s),\
                                   tf.scalar_mul((1-self.alpha_pred),cross_entropy_c) )
                  
            train_step = tf.train.AdamOptimizer(0.00006).minimize(cross_entropy)
            
            #prediction = tf.nn.softmax(output_layer_s)     #Predctions for the training
            
        #graph architecture end####
        
        def accuracy(predictions, labels):
            return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                    / predictions.shape[0])           
        
        mini_batches_x_valid = [ valid_x[k:k+self.mini_batch_size_classifier] for k in range(0, valid_len, self.mini_batch_size_classifier) ]
        mini_batches_y_valid = [ valid_y[k:k+self.mini_batch_size_classifier] for k in range(0, valid_len, self.mini_batch_size_classifier) ]    
        
        # train on mini batches
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())     #Initializing all the graph variables
            saver = tf.train.Saver()                        #For saving the model after training
            
            print("->Training started at ", time.ctime(time.time()) )
            for epo in range(self.classifier_epochs):
                #Training
                #Re-shuffling the training data for each epoch to improve generalization
                temp = list(zip(train_x, train_y, train_x_noise))
                random.shuffle(temp) 
                train_x, train_y, train_x_noise = zip(*temp)

                mini_batches_x_train        = [ train_x[k:k+self.mini_batch_size_classifier] for k in range(0, train_len, self.mini_batch_size_classifier) ]
                mini_batches_x_noise_train  = [ train_x_noise[k:k+self.mini_batch_size_classifier] for k in range(0, train_len, self.mini_batch_size_classifier) ]
                mini_batches_y_train        = [ train_y[k:k+self.mini_batch_size_classifier] for k in range(0, train_len, self.mini_batch_size_classifier) ]
                  
                train_acc_sum = 0
                train_loss_sum = 0
                for mini_count in range(len(mini_batches_x_train)):
                    batch_x       = mini_batches_x_train[mini_count]
                    batch_x_noise = mini_batches_x_noise_train[mini_count]
                    batch_y       = mini_batches_y_train[mini_count]  
    
                    one1 = np.ones(len(batch_y))
                    one1 = one1.reshape(len(one1),1)
                    
                    feed_dict={X: batch_x, X_noise: batch_x_noise, Y: batch_y, Y_c:one1}
                    _, mini_loss, predictions = sess.run([train_step, cross_entropy, output_layer_s], feed_dict)
                    train_loss_sum += mini_loss
                    train_acc_sum += accuracy(predictions, batch_y)
                    
                train_loss = train_loss_sum/len(mini_batches_x_train)    
                train_acc  = train_acc_sum/len(mini_batches_x_train)
                
                #Validation
                valid_acc_sum = 0
                for mini_count in range(len(mini_batches_x_valid)):
                    batch_x = mini_batches_x_valid[mini_count]
                    batch_y = mini_batches_y_valid[mini_count]  
                    
                    feed_dict={X: batch_x, Y: batch_y}
                    #predictions = sess.run(prediction, feed_dict)
                    predictions = sess.run(output_layer_s, feed_dict)
                    valid_acc_sum  += accuracy(predictions, batch_y)
                    
                valid_acc = valid_acc_sum/len(mini_batches_x_valid)        
                
                print("\n-->Epoch", epo+1, " completed at ",time.ctime(time.time()) )                
                print("\tTrain loss = {:.2f}\tTrain acc = {:.2f}\tValid acc = {:.2f}".format(train_loss, train_acc, valid_acc))
                
            print("->Classifier training completed at ", time.ctime(time.time()))
            
            print('->Saving model...')
            saver.save(sess, './saved_model/my-model', global_step =2)  
        
#DRIVER CODE
if __name__=='__main__':
    np.random.seed(0)
    
    #Feed the name (with path, if in a different folder) of the data files in csv format.
    #Each row is in the following format : [label, mail]
    all_data_file = 'generated_dataset.csv' 
    training_data_file = 'training_data.csv'
    
    #It is executed before training the classifier.
    #Once the skipgrams are generated, it can be commented for later tuning the hyper-parameters of the classifier only.
    print("Generating skipgrams")
    skipgram_generator(all_data_file)   
    print("Skipgrams generated")
    
    #hyperparameter configuration for training classifier
    config={'m':20, 'window_size':7, 'mini_batch_size_classifier':256, 'classifier_epochs':4, \
            'alpha_pred':0.8, 'k':25}   #hyper-parameters
     
    model = NeuralNet(config)
    
    print("Generating corpus...", time.ctime(time.time()))    
    filtered_mails, polarity= model.create_corpus(training_data_file)
    
    print("Generating classifier training_data...", time.ctime(time.time()))
    x, y, x_noise = model.classifier_dataset(filtered_mails, polarity)      

    print("Training classifier...", time.ctime(time.time()))
    model.train_classifier(x, y, x_noise)      #training classifier neural network