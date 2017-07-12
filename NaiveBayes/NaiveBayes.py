
# coding: utf-8

# # Naive Bayes Model
# 
#  #### P(A|B) = P(AnB) / P(B)
#  #### P(AnB) = P(A|B) / P(B)
#  #### P(B|A) * P(A) = P(A|B) * P(B)
#  #### P(B|A) = P(A|B) * P(B) / P(A)

# In[1]:


import re
from collections import defaultdict
import pandas as pd
import numpy as np 
import os
import math

#csv 파일 불러옵니다. 데이터셋은 keggle의 무료데이터셋을 이용하였습니다.
load_csv = pd.read_csv(os.getcwd()+'/dataset/spam.csv',encoding='latin-1')

#데이터 형식이 어떻게 이루어져 있는지 상위 10개를 꺼내와 확인합니다.
load_csv.head(10)


# In[2]:


#필요없는  column (Unnamed:2 , Unnamed: 3 , Unnamed: 4) 삭제합니다. 
load_csv = load_csv.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

#라벨형식을 스팸일때 0 햄일때 1 으로 바꾸어 줍니다.
load_csv["v1"]=load_csv["v1"].apply(lambda x:1 if x=="spam" else 0) 

#column 이름 재지정
load_csv = load_csv.rename(columns = {'v1':'Class','v2':'Text'})

#cloum 위치 변경
load_csv = pd.DataFrame(load_csv, columns=['Text', 'Class'])

#데이터 랜덤하게 섞어줍니다. 
load_csv = load_csv.sample(frac=1).reset_index(drop=True)

#데이터를 [['메일 내용' , '0 or 1' ] , ['메일내용' , '0  or 1']] 의 형식으로 바꾸어 줍니다.
text = load_csv['Text']
text = list(text)

label = load_csv['Class']
label = list(label)

data = [x for x in zip(text,label)]

# 학습데이터와 테스트데이터를 7:3 의 비율로 나누어 줍니다. 
train_data = data[:int(len(data)*0.7)]
test_data = data[int(len(data)*0.7):]

#데이터 형식을 알아보기위해 학습데이터의 shape 을 확인합니다.
print(train_data[:3])


# In[3]:


#문장을 단어로 분해해 주는 함수를 만듭니다. [a-z0-9]+ 는 정규식입니다.
def tokenize(message):
    message = message.lower()
    all_words = re.findall("[a-z0-9]+",message)
    return set(all_words)


# (단어)  : [  스팸메세지에서 나온 빈도수] [스팸이 아닌 메시지에서 나온 빈도수]
def count_words(training_set):
    counts = defaultdict(lambda:[0,0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
            
    return counts

# [단어, 스팸 메시지에서 단어가 나올 확률, 스팸이 아닌 메시지에서 단어가 나올 확률]
# k = 0.5 는 가짜 빈도수를 뜻합니다 . 가짜 빈도수를 사용 안할경우 : 만일 ham 메일에서만 '나비' 라는
#단어가 나왔을 때  spam 에서 '나비모양의 가방을 싸게 판매합니다 ' 의 메일도 ham메일로 예측하기 때문에
#가짜 빈도수를 넣어줍니다.
def word_probabilities(counts, total_spams, total_non_spam,k=0.5):
    return [(w,
            (spam+k) / (total_spams + 2*k),
            (non_spam + k) / (total_non_spam + 2*k))
           for w , (spam,non_spam) in counts.items()]

#단어의 확률을 사용해서 메세지가 스팸일 확률을 계산합니다.
def spam_probability(word_probs , message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0
    
    #모든 단어에 대해서 반복합니다.
    for word , prob_if_spam, prob_if_not_spam in word_probs:
        
        #만약에 메세지에 word 가 나타나면 해당 단어가 나올 log  확률을 더해줍니다.
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)
        else:
            log_prob_if_spam += math.log(1- prob_if_spam)
            log_prob_if_not_spam += math.log(1 - prob_if_not_spam)
    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

#최종적으로 만들어 놓은 함수들을 이용하여 나이브 베이즈 모델을 만듭니다.
class NaiveBayes:
    def __init__(self,k=0.5):
        self.k = k
        self.word_prob = []
        
    def train(self,training_set):
        # 트레이닝 데이터에서 스팸메일의 개수
        num_spams = len([is_spam for message, is_spam in training_set if is_spam])
       # 트레이닝 데이터에서 햄메일의 개수
        num_non_spams = len(training_set)-num_spams
        
        word_counts = count_words(training_set)
        
        self.word_probs = word_probabilities(word_counts,num_spams,num_non_spams,self.k)
    def classify(self,message):
        return spam_probability(self.word_probs,message)
        
classifier = NaiveBayes()
classifier.train(train_data)

classified = [(subject, is_spam ,classifier.classify(subject))
             for subject, is_spam in test_data]

#스팸일 확률이 50% 넘을 때 스팸메일의 내용과 확률을 출력하기 위해 사용됩니다.
for i in range(len(classified)):
    if classified[i][2]>0.5:
        print(classified[i])


# -결과
# 
#  *출력된 결과는 테스트 데이터중 스팸메일의 경우 출력되어 스팸메일의 내용과 확률을 보여줍니다.
