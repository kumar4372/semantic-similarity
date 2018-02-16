import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.wsd import lesk
import numpy as np
from scipy.optimize import linear_sum_assignment
from nltk import  pos_tag, ne_chunk
import nltk.tag.stanford as st
classifier='/home/gautam/Desktop/Courses/MTL785/project/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz'
jar='/home/gautam/Desktop/Courses/MTL785/project/stanford-ner-2017-06-09/stanford-ner.jar'
s=st.StanfordNERTagger(classifier,jar)
# nltk.download('wordnet')
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
stop_word = set(stopwords.words('english'))

df = pd.read_csv('data/train.csv')
# print(df.columns.values)
question2_total = df.iloc[:,4].values
question1_total = df.iloc[:,3].values
# question1_total = ['what is your name']
# question2_total = ['what should I call you']
# print(question1_total)
question1 = word_tokenize(question1_total[0])
question2 = word_tokenize(question2_total[0])
print(question1)
print(question2)


nerq1=s.tag(question1)

nerq2=s.tag(question2)
print("##########")
for i in nerq1:
	# print(i[1])
	if(i[1]=="LOCATION"):
		loc1=i[0]
	if(i[1]=="NAME"):
		name1=i[0]
print(loc1)


for i in nerq2:
	# print(i[1])
	if(i[1]=="LOCATION"):
		loc2=i[0]
	if(i[1]=="NAME"):
		name2=i[0]
# print(nerq2)
print("##########")

tagged1 = nltk.pos_tag(question1)
tagged2 = nltk.pos_tag(question2)
# filtered_q1 = [w for w in tagged1 if not w[0] in stop_word]
# filtered_q2 = [w for w in tagged2 if not w[0] in stop_word]
# q1 = [(stemmer.stem(w[0]),w[1]) for w in tagged1] 
# q2 = [(stemmer.stem(w[0]),w[1]) for w in tagged2]

# print(tagged1)
# print(tagged2)
common_words = [word for word in question1 if word in question2]
# print(common_words)

# list = []
list1 = []
list2 = []
for item in tagged1:
	list1.append(lesk(tagged1,item[0]))
	# print(item[0])
	# print(item[1])
print(list1)
for item in tagged2:
	list2.append(lesk(tagged2,item[0]))
	# print(item[0])
	# print(item[1])
print(list2)

R = np.zeros((len(list1),len(list2)))
for i in range(len(list1)):
	for j in range(len(list2)):
		if list1[i]and list2[j]:
			R[i][j] = list1[i].wup_similarity(list2[j])
print(R)
# row_ind, col_ind = linear_sum_assignment(R)
# print(row_ind)
# print(col_ind)
# for word1 in question1:
#     for word2 in question2:
#         wordFromList1 = wordnet.synsets(word1)
#         wordFromList2 = wordnet.synsets(word2)
#         # print(wordFromList1)
#         # print(wordFromList2)
#         if wordFromList1 and wordFromList2: #Thanks to @alexis' note
#             s = wordFromList1[0].wup_similarity(wordFromList2[0])
#             # print()
#             if (s>0.8)
#             	print(s)
#             	print(word1)
#             	print(word2)
			# list.append(s)
			# break
	# break
# print(list)
# print(max(list))
# print(lesk())