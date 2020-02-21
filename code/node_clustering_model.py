import random
import string
import re
import numpy
from itertools import *
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import csv
import argparse

parser = argparse.ArgumentParser(description = 'link prediction task')
parser.add_argument('--C_n', type = int, default = 4,
			   help = 'number of node class label')
parser.add_argument('--data_path', type = str, default = '../data/academic_test/',
				   help='path to data')
parser.add_argument('--embed_d', type = int, default = 128,
			   help = 'embedding dimension')

args = parser.parse_args()
print(args)


def model(cluster_id_num):
	cluter_embed = numpy.around(numpy.random.normal(0, 0.01, [cluster_id_num, args.embed_d]), 4)
	cluster_embed_f = open(args.data_path + "cluster_embed.txt", "r")
	for line in cluster_embed_f:
		line=line.strip()
		author_index=int(re.split(' ',line)[0])
		embed_list=re.split(' ',line)[1:]
		for i in range(len(embed_list)):
			cluter_embed[author_index][i] = embed_list[i]

	kmeans = KMeans(n_clusters = args.C_n, random_state = 0).fit(cluter_embed) 

	cluster_id_list = [0] * cluster_id_num
	cluster_id_f = open(args.data_path + "cluster.txt", "r")
	for line in cluster_id_f:
		line = line.strip()
		author_index = int(re.split(',',line)[0])
		cluster_id = int(re.split(',',line)[1])
		cluster_id_list[author_index] = cluster_id

	#NMI
	print ("NMI: " + str(normalized_mutual_info_score(kmeans.labels_, cluster_id_list)))
	print ("ARI: " + str(adjusted_rand_score(kmeans.labels_, cluster_id_list)))

