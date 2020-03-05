import re, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# stops = set(stopwords.words('english'))
stops = '[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
stopsnotremoved = '.-#+'
stopsremoved = '#!"$%&\'()<>*,/:=?@[]\\^_`{|}~;,'
links = '((www\.\S+)|(https?://\S+))'
linksreg = re.compile(links)

clicpattern = [
(r'won\'t', 'will not'),
(r'can\'t', 'can not'),
(r'i\'m', 'i am'),
(r'ain\'t', 'is not'),
(r'(\w+)\'ll', '\g<1> will'),
(r'(\w+)n\'t', '\g<1> not'),
(r'(\w+)\'ve', '\g<1> have'),
(r'(\w+)\'s', '\g<1> is'),
(r'(\w+)\'re', '\g<1> are'),
(r'(\w+)\'d', '\g<1> would'),
(r'</?code>', ' '),
(r'\\n', ' '),
(r'</?a>', ' '),
(r'href', ' '),
(r'</?p>', ' ')]
class clitic(object):
	def __init__(self, patterns=clicpattern):
		self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
	
	def replace(self, text):
		s = text
		for (pattern, repl) in self.patterns:
			(s, count) = re.subn(pattern, repl, s)
		return s
rep=clitic()

def stopwordhandle(st):
	global stopsremoved
	# st = stopreg.sub(' ',st)
	for w in stopsremoved:
		st = st.replace(w, ' ')
	return st

def preprocessBody(st):
	# global stopreg
	# global linksreg
	#Convert to lower case
	# st = st.lower()

	#Convert urls to LINK
	# st = linksreg.sub(' LINK ',st)
	# st = re.sub('((www\.\S+)|(https?://\S+))',' LINK ',st)


	#Convert 3 or more repetitions of character to 2 repetitions of same character
	# st = re.sub(r'(.)\1{2,}',r'\1\1',st,re.DOTALL)

	# clitisization
	# st = re.sub(r"n\'t",r' not', st)
	# st = re.sub(r"\'m",r' am', st)
	# st = re.sub(r"\'ve",r' have', st)
	# st = re.sub(r"\'re",r' are', st)
	# st = re.sub(r"\'s",r'', st)
#	rep.replace(st)

	# Strip punctuation
	# st = st.strip(string.punctuation)
	# st = stopreg.sub(' \g<1> ',st)


	#Convert multiple white spaces into a single space
	# st = re.sub('\s+', ' ', st)


	# alltokens = word_tokenize(st)
	
	# # remove stopwords
	# tokensNew = []
	# for xx in alltokens:
	# 	if xx not in stops:
	# 		tokensNew.append(xx)

	# handle negations
	# tokens = []
	# neg = 0 #neg is a counter that decrements and becomes 0
	# nStr = "(?:^(?:never|no|nothing|cannot|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|neither|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)"
	# for xx in alltokens:
	# 	newxx = "not"+xx if (neg>=1) else xx
	# 	tokens.append(newxx)
	# 	if (neg>=1): neg -= 1
	# 	if (re.search(nStr,xx)):
	# 		neg = 2

	# st = " ".join(tokensNew)

	return st

def preprocessTitle(st):
	return preprocessBody(st)
