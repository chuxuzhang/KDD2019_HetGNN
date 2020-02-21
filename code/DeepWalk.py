import string;
import re;
import random
import math
import numpy as np
from gensim.models import Word2Vec
from itertools import *
dimen = 128
window = 5


def read_random_walk_corpus():
	walks=[]
	#inputfile = open("../data/academic_test/meta_random_walk_APVPA_test.txt","r")
	inputfile = open("../data/academic_test/het_random_walk_test.txt", "r")
	for line in inputfile:
		path = []
		node_list=re.split(' ',line)
		for i in range(len(node_list)):
			path.append(node_list[i])			
		walks.append(path)
	inputfile.close()
	return walks


walk_corpus = read_random_walk_corpus()
model = Word2Vec(walk_corpus, size = dimen, window = window, min_count = 0, workers = 2, sg = 1, hs = 0, negative = 5)


print("Output...")
#model.wv.save_word2vec_format("../data/node_embedding.txt")
model.wv.save_word2vec_format("../data/academic_test/node_net_embedding.txt")

