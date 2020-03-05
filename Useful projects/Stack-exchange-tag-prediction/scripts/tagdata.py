import random, csv, cPickle, os
import pandas

numtags = 80
with open(os.path.join('data', 'numtags'), 'w') as fid:
    cPickle.dump(numtags, fid)

with open(os.path.join('data', 'tagsmall'), 'r') as fid:
    tags_count = cPickle.load(fid)

with open(os.path.join('data', 'trainsmall'), 'r') as fid:
    randomdata = cPickle.load(fid)
    # count = cPickle.load(fid)

sorted_tags = sorted(((len(v), k) for k, v in tags_count.items()), reverse=True)
sorted_tags = sorted_tags[:numtags]

datacount = len(randomdata)
total = set(range(datacount))
j=0
for v,k in sorted_tags:
	print j
	l = random.sample(tags_count[k],70)
	pos = set(l)
	neg = total - pos
	m = random.sample(list(neg),1510)
	positive = [randomdata[i][1:3] for i in l]
	negative = [randomdata[i][1:3] for i in m]
	with open(os.path.join('data', str(j)+"pos"), 'w') as fid:
		cPickle.dump(positive, fid)

	with open(os.path.join('data', str(j)+"neg"), 'w') as fid:
		cPickle.dump(negative, fid)
	j = j+1
