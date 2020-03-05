# coding: utf-8
from gensim.models import Word2Vec
import cPickle, numpy
from gensim import utils

dataloc = "data/"
dataFile = dataloc+"trainsmall"
outfile = "data/words"
minfreq = 5

with open(dataFile,'r') as ff:
	fulldata = cPickle.load(ff)

words = {}

for i in xrange(len(fulldata)):
	# fulldata[i][1] = p.preprocessTitle(fulldata[i][1])+" "+p.preprocessBody(fulldata[i][2])
	fulldata[i][1] = (fulldata[i][1])+" "+(fulldata[i][2])
	# temp = (utils.to_unicode(fulldata[i][1])).split()
	temp = fulldata[i][1].split()
	for wd in temp:
		if (len(wd)<3):
			continue
		if wd not in words:
			words[wd] = 1
		else:
			words[wd]+=1

# print "Loading Model"
# model = Word2Vec.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
# print "Model Loaded"

ans = []
for key,val in words.iteritems():
	# atleast 3 length words with freq >= 8
	if (val>=minfreq):
		ans.append((key,val))

ans = sorted(ans,key = lambda x: x[1],reverse=True)
ans = ans[:50000]

with open(outfile,'w') as ff:
	cPickle.dump(ans,ff)