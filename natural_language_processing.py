# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 02:24:54 2017

@author: Mahsa
"""
import csv

from textblob.classifiers import NaiveBayesClassifier
#from textblob import TextBlob
#from sklearn.naive_bayes import GaussianNB

train=[]
test=[]        #Array Definition
path1 =  r'D:\1\training_data.csv'     #Address Definition

path11 =  r'D:\test_data.csv' 
with open(path1, 'r', encoding="utf8") as f1:    #Open File as read by 'r'
    reader=[tuple(line[col] for col in (1,2)) for line in csv.reader(f1)]
 	
print(reader)
NB = NaiveBayesClassifier(reader)
print('it has finished..')
commonproductname=[]
Probability=[]

path11 =  r'D:\test_data_1.csv' 
with open(path11, 'r', encoding="utf8") as f11:    #Open File as read by 'r'
    reader11 = csv.reader(f11)  
    next(reader11, None)          #Skip header because file header is not needed
    for row11 in reader11:          #fill array by file info by for loop
        test.append(row11[0])

for i in range(0,len(test)):
    prob_dist = NB.prob_classify(test[i])
    commonproductname.append(prob_dist.max())
    Probability.append(prob_dist.prob(prob_dist.max()))
    
print(commonproductname)
print(Probability)