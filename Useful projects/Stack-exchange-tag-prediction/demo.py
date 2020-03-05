import csv, cPickle, nltk, re, sys, random, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from nltk.stem import *
import scripts.preprocessing as p
from gensim.models import Doc2Vec
from gensim  import utils
from scipy import sparse

# d2vdata = Doc2Vec.load('./stackovf300.d2v')

## Prediction ##
# fname = "data/"
# with open(fname, 'r') as f:
#     csv_reader = csv.reader(f)
#     i=-1
#     for row in csv_reader:
#     	i=i+1
#     	if exist.has_key(i):
#     		randomdata.append(row)
#     	else:
#     		continue

with open(os.path.join('data', 'testdata'),'r') as ff:
	testdata = cPickle.load(ff)

with open(os.path.join('models', 'vocab'),'r') as ff:
	tfdifVect = cPickle.load(ff)

with open(os.path.join('data', 'numtags'),'r') as ff:
	numtags = cPickle.load(ff)

print "Data Read"
tag_names = {}
clf = {}
for i in xrange(numtags):
	print i
	with open(os.path.join('models', str(i)+"clf"),'r') as ff:
		clf[i] = cPickle.load(ff)
		tagname = cPickle.load(ff)
	tag_names[i] = tagname

while True:
	print "Enter file name"
	filename = raw_input()
	f = open(filename,'r')
	lines = f.readlines()
	f.close()
	questitle = lines[0]
	quesbody = ' '.join(lines[1:])
	testScore = []

	# print questitle
	# print quesbody
	questext = p.preprocessTitle(questitle)+" "+ p.preprocessBody(quesbody)
	print questext

	testDataVect = tfdifVect.transform([questext])
	for i in xrange(numtags):
		tagTestScore= clf[i].predict_log_proba(testDataVect)
		testScore = testScore + [(tagTestScore[0][1], i)]

	sorted_tags = sorted(testScore, reverse=True)
	predicted_tags = [tag_names[b] for (a,b) in sorted_tags[:5]]
	print predicted_tags
# print sorted_tags[:5]

