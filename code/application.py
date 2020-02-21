import string
import re
import numpy as np
import os
import sys
import random
from itertools import *
import argparse
import link_prediction_model as LP
import node_classification_model as NC
import node_clustering_model as NCL


parser = argparse.ArgumentParser(description = 'application data process')
parser.add_argument('--A_n', type = int, default = 28646,
			   help = 'number of author node')
parser.add_argument('--P_n', type = int, default = 21044,
			   help = 'number of paper node')
parser.add_argument('--V_n', type = int, default = 18,
			   help = 'number of venue node')
parser.add_argument('--C_n', type = int, default = 4,
			   help = 'number of node class label')
parser.add_argument('--data_path', type = str, default = '../data/academic_test/',
				   help='path to data')
parser.add_argument('--embed_d', type = int, default = 128,
			   help = 'embedding dimension')

args = parser.parse_args()
print(args)


def a_a_collab_feature_setting():
	a_embed = np.around(np.random.normal(0, 0.01, [args.A_n, args.embed_d]), 4)
	embed_f = open(args.data_path + "node_embedding.txt", "r")
	for line in islice(embed_f, 0, None):
		line = line.strip()
		node_id = re.split(' ', line)[0]
		if len(node_id) and (node_id[0] in ('a', 'p', 'v')):
			type_label = node_id[0]
			index = int(node_id[1:])
			embed = np.asarray(re.split(' ',line)[1:], dtype='float32')
			if type_label == 'a':
				a_embed[index] = embed
	embed_f.close()

	train_num = 0
	a_a_list_train_f = open(args.data_path + "a_a_list_train.txt", "r")
	a_a_list_train_feature_f = open(args.data_path + "train_feature.txt", "w")
	for line in a_a_list_train_f:
		line = line.strip()
		a_1 = int(re.split(',',line)[0])
		a_2 = int(re.split(',',line)[1])
		label = int(re.split(',',line)[2])
		if random.random() < 0.2:#training data ratio 
			train_num += 1
			a_a_list_train_feature_f.write("%d, %d, %d,"%(a_1, a_2, label))
			for d in range(args.embed_d - 1):
				a_a_list_train_feature_f.write("%f,"%(a_embed[a_1][d] * a_embed[a_2][d]))
			a_a_list_train_feature_f.write("%f"%(a_embed[a_1][args.embed_d - 1] * a_embed[a_2][args.embed_d - 1]))
			a_a_list_train_feature_f.write("\n")
	a_a_list_train_f.close()
	a_a_list_train_feature_f.close()
	#print train_num

	test_num = 0
	a_a_list_test_f = open(args.data_path + "a_a_list_test.txt", "r")
	a_a_list_test_feature_f = open(args.data_path + "test_feature.txt", "w")
	for line in a_a_list_test_f:
		line = line.strip()
		a_1 = int(re.split(',',line)[0])
		a_2 = int(re.split(',',line)[1])
		label = int(re.split(',',line)[2])
		test_num += 1
		a_a_list_test_feature_f.write("%d, %d, %d,"%(a_1, a_2, label))
		for d in range(args.embed_d - 1):
			a_a_list_test_feature_f.write("%f,"%(a_embed[a_1][d] * a_embed[a_2][d]))
		a_a_list_test_feature_f.write("%f"%(a_embed[a_1][args.embed_d - 1] * a_embed[a_2][args.embed_d - 1]))
		a_a_list_test_feature_f.write("\n")
	a_a_list_test_f.close()
	a_a_list_test_feature_f.close()
	
	# print("a_a_train_num: " + str(train_num))
	# print("a_a_test_num: " + str(test_num))

	return train_num, test_num


def a_p_cite_feature_setting():
	a_embed = np.around(np.random.normal(0, 0.01, [args.A_n, args.embed_d]), 4)
	p_embed = np.around(np.random.normal(0, 0.01, [args.P_n, args.embed_d]), 4)
	embed_f = open(args.data_path + "node_embedding.txt", "r")
	for line in islice(embed_f, 0, None):
		line = line.strip()
		node_id = re.split(' ', line)[0]
		if len(node_id) and (node_id[0] in ('a', 'p', 'v')):
			type_label = node_id[0]
			index = int(node_id[1:])
			embed = np.asarray(re.split(' ',line)[1:], dtype='float32')
			if type_label == 'a':
				a_embed[index] = embed
			elif type_label == 'p':
				p_embed[index] = embed
	embed_f.close()

	train_num = 0
	a_p_cite_list_train_f = open(args.data_path + "a_p_cite_list_train.txt", "r")
	a_p_cite_list_train_feature_f = open(args.data_path + "train_feature.txt", "w")
	for line in a_p_cite_list_train_f:
		line = line.strip()
		a_1 = int(re.split(',',line)[0])
		p_2 = int(re.split(',',line)[1])
		label = int(re.split(',',line)[2])
		if random.random() < 0.2:#training data ratio 
			train_num += 1
			a_p_cite_list_train_feature_f.write("%d, %d, %d,"%(a_1, p_2, label))
			for d in range(args.embed_d - 1):
				a_p_cite_list_train_feature_f.write("%f,"%(a_embed[a_1][d] * p_embed[p_2][d]))
			a_p_cite_list_train_feature_f.write("%f"%(a_embed[a_1][args.embed_d - 1] * p_embed[p_2][args.embed_d - 1]))
			a_p_cite_list_train_feature_f.write("\n")
	a_p_cite_list_train_f.close()
	a_p_cite_list_train_feature_f.close()
	#print train_num

	test_num = 0
	a_p_cite_list_test_f = open(args.data_path + "a_p_cite_list_test.txt", "r")
	a_p_cite_list_test_feature_f = open(args.data_path + "test_feature.txt", "w")
	for line in a_p_cite_list_test_f:
		line = line.strip()
		a_1 = int(re.split(',',line)[0])
		p_2 = int(re.split(',',line)[1])
		label = int(re.split(',',line)[2])
		test_num += 1
		a_p_cite_list_test_feature_f.write("%d, %d, %d,"%(a_1, p_2, label))
		for d in range(args.embed_d - 1):
			a_p_cite_list_test_feature_f.write("%f,"%(a_embed[a_1][d] * p_embed[p_2][d]))
		a_p_cite_list_test_feature_f.write("%f"%(a_embed[a_1][args.embed_d - 1] * p_embed[p_2][args.embed_d - 1]))
		a_p_cite_list_test_feature_f.write("\n")
	a_p_cite_list_test_f.close()
	a_p_cite_list_test_feature_f.close()
	
	# print("a_p_cite_train_num: " + str(train_num))
	# print("a_p_cite_test_num: " + str(test_num))

	return train_num, test_num


def a_v_recommendation():
	a_embed = np.around(np.random.normal(0, 0.01, [args.A_n, args.embed_d]), 4)
	v_embed = np.around(np.random.normal(0, 0.01, [args.V_n, args.embed_d]), 4)
	embed_f = open(args.data_path + "node_embedding.txt", "r")
	for line in islice(embed_f, 0, None):
		line = line.strip()
		node_id = re.split(' ', line)[0]
		if len(node_id) and (node_id[0] in ('a', 'p', 'v')):
			type_label = node_id[0]
			index = int(node_id[1:])
			embed = np.asarray(re.split(' ',line)[1:], dtype='float32')
			if type_label == 'a':
				a_embed[index] = embed
			elif type_label == 'v':
				v_embed[index] = embed
	embed_f.close()

	a_v_list_train = [[] for k in range(args.A_n)]
	a_v_list_test = [[] for k in range(args.A_n)]
	v_train = [0] * args.V_n
	a_v_list_f = ["a_v_list_train.txt", "a_v_list_test.txt"]
	for i in range(len(a_v_list_f)):
		f_name = a_v_list_f[i]
		neigh_f = open(args.data_path + f_name, "r")
		for line in neigh_f:
			line = line.strip()
			node_id = int(re.split(':', line)[0])
			neigh_list = re.split(':', line)[1]
			neigh_list_id = re.split(',', neigh_list)
			if f_name == 'a_v_list_train.txt':
				for j in range(len(neigh_list_id)-1):
					v_train[int(neigh_list_id[j])] = 1
					a_v_list_train[node_id].append(int(neigh_list_id[j]))
			else:
				for j in range(len(neigh_list_id)-1):
					a_v_list_test[node_id].append(int(neigh_list_id[j]))
		neigh_f.close()

	topK = 3
	topK_2 = 5

	a_num =0 
	recall_all = 0
	recall_all_2 = 0
	pre_all = 0
	pre_all_2 = 0
	F_1 = 0 
	F_2 = 0

	for i in range(args.A_n):
		if len(a_v_list_test[i]):
			a_num += 1
			score_list = []
			correct_pair = 0
			correct_pair_2 = 0

			for j in range(len(a_v_list_test[i])):
				score_list.append(np.dot(a_embed[i], v_embed[a_v_list_test[i][j]]))

			for jj in range(args.V_n):
				if jj not in a_v_list_test[i] and jj not in a_v_list_train[i] and v_train[jj]:
					score_list.append(np.dot(a_embed[i], v_embed[jj]))

			sort_index = np.ndarray.tolist(np.argsort(score_list))
			score_list.sort()

			score_threshold = score_list[-topK - 1]
			score_threshold_2 = score_list[-topK_2 - 1]

			venue_num_temp = 0
			for j in range(len(a_v_list_test[i])):
				venue_num_temp += 1
				score_temp = np.dot(a_embed[i], v_embed[int(a_v_list_test[i][j])])
				
				if score_temp > score_threshold:
					correct_pair += 1
				if score_temp > score_threshold_2:
					correct_pair_2 += 1

			recall_all += float(correct_pair) / len(a_v_list_test[i])
			recall_all_2 += float(correct_pair_2) / len(a_v_list_test[i])

			pre_all += float(correct_pair) / topK
			pre_all_2 += float(correct_pair_2) / topK_2

	#print("total_author: " + str(a_num))
	recall_all =  recall_all / a_num
	recall_all_2 =  recall_all_2 / a_num
	pre_all =  pre_all / a_num
	pre_all_2 =  pre_all_2 / a_num
	F_1= (2*recall_all*pre_all)/(recall_all + pre_all)
	F_2= (2*recall_all_2*pre_all_2)/(recall_all_2 + pre_all_2)
	print("recall@top3: "+str(recall_all))
	#print("recall@top5: "+str(recall_all_2))
	print("pre@top3: "+str(pre_all))
	#print("pre@top5 "+str(pre_all_2))
	print("F1@top3: "+str(F_1))
	#print("F1@top5: "+str(F_2))


def a_class_cluster_feature_setting():
	a_embed = np.around(np.random.normal(0, 0.01, [args.A_n, args.embed_d]), 4)
	embed_f = open(args.data_path + "node_embedding.txt", "r")
	for line in islice(embed_f, 0, None):
		line = line.strip()
		node_id = re.split(' ', line)[0]
		if len(node_id) and (node_id[0] in ('a', 'p', 'v')):
			type_label = node_id[0]
			index = int(node_id[1:])
			embed = np.asarray(re.split(' ',line)[1:], dtype='float32')
			if type_label == 'a':
				a_embed[index] = embed
	embed_f.close()

	a_p_list_train = [[] for k in range(args.A_n)]
	a_p_list_train_f = open(args.data_path + "a_p_list_train.txt", "r")
	for line in a_p_list_train_f:
		line = line.strip()
		node_id = int(re.split(':', line)[0])
		neigh_list = re.split(':', line)[1]
		neigh_list_id = re.split(',', neigh_list)
		for j in range(len(neigh_list_id)):
			a_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
	a_p_list_train_f.close()

	p_v = [0] * args.P_n
	p_v_f = open(args.data_path + 'p_v.txt', "r")
	for line in p_v_f:
		line = line.strip()
		p_id = int(re.split(',',line)[0])
		v_id = int(re.split(',',line)[1])
		p_v[p_id] = v_id
	p_v_f.close()

	a_v_list_train = [[] for k in range(args.A_n)]
	for i in range(len(a_p_list_train)):#tranductive node classification
		for j in range(len(a_p_list_train[i])):
			p_id = int(a_p_list_train[i][j][1:])
			a_v_list_train[i].append(p_v[p_id])

	a_v_num = [[0 for k in range(args.V_n)] for k in range(args.A_n)]
	for i in range(args.A_n):
		for j in range(len(a_v_list_train[i])):
			v_index = int(a_v_list_train[i][j])
			a_v_num[i][v_index] += 1

	a_max_v = [0] * args.A_n
	for i in range(args.A_n):
		a_max_v[i] =  a_v_num[i].index(max(a_v_num[i]))

	cluster_f = open(args.data_path + "cluster.txt", "w")
	cluster_embed_f = open(args.data_path + "cluster_embed.txt", "w")
	a_class_list = [[] for k in range(args.C_n)]
	cluster_id = 0
	num_hidden = args.embed_d
	for i in range(args.A_n):
		if len(a_p_list_train[i]):
			if a_max_v[i] ==17 or a_max_v[i] == 4 or a_max_v[i] == 1:#cv: cvpr, iccv, eccv
				a_class_list[0].append(i)
				cluster_f.write("%d,%d\n"%(cluster_id,3))
				cluster_embed_f.write("%d "%(cluster_id))
				for k in range(num_hidden):
					cluster_embed_f.write("%lf "%(a_embed[i][k]))
				cluster_embed_f.write("\n")
				cluster_id += 1
			elif a_max_v[i] == 16 or a_max_v[i] == 2 or a_max_v[i] == 3: #nlp: acl, emnlp, naacl
				a_class_list[1].append(i)
				cluster_f.write("%d,%d\n"%(cluster_id,0))
				cluster_embed_f.write("%d "%(cluster_id))
				for k in range(num_hidden):
					cluster_embed_f.write("%lf "%(a_embed[i][k]))
				cluster_embed_f.write("\n")
				cluster_id += 1
			elif a_max_v[i] == 9 or a_max_v[i] == 13 or a_max_v[i] == 6: #dm: kdd, wsdm, icdm
				a_class_list[2].append(i)
				cluster_f.write("%d,%d\n"%(cluster_id,1))
				cluster_embed_f.write("%d "%(cluster_id))
				for k in range(num_hidden):
					cluster_embed_f.write("%lf "%(a_embed[i][k]))
				cluster_embed_f.write("\n")
				cluster_id += 1
			elif a_max_v[i] == 12 or a_max_v[i] == 10 or a_max_v[i] == 5: #db: sigmod, vldb, icde
				a_class_list[3].append(i)
				cluster_f.write("%d,%d\n"%(cluster_id,2))
				cluster_embed_f.write("%d "%(cluster_id))
				for k in range(num_hidden):
					cluster_embed_f.write("%lf "%(a_embed[i][k]))
				cluster_embed_f.write("\n")
				cluster_id += 1
	cluster_f.close()
	cluster_embed_f.close()
	#print id_

	a_class_train_f = open(args.data_path + "a_class_train.txt", "w")
	a_class_test_f = open(args.data_path + "a_class_test.txt", "w")
	train_class_feature_f = open(args.data_path + "train_class_feature.txt", "w")
	test_class_feature_f = open(args.data_path + "test_class_feature.txt", "w")
	train_num = 0
	test_num = 0
	for i in range(args.C_n):
		for j in range(len(a_class_list[i])):
			randvalue = random.random()
			if randvalue < 0.1:
				a_class_train_f.write("%d,%d\n"%(a_class_list[i][j],i))
				train_class_feature_f.write("%d,%d," %(a_class_list[i][j],i))
				for d in range(num_hidden - 1):
					train_class_feature_f.write("%lf," %a_embed[a_class_list[i][j]][d])
				train_class_feature_f.write("%lf" %a_embed[a_class_list[i][j]][num_hidden-1])
				train_class_feature_f.write("\n")
				train_num += 1
			else:
				a_class_test_f.write("%d,%d\n"%(a_class_list[i][j],i))
				test_class_feature_f.write("%d,%d," %(a_class_list[i][j],i))
				for d in range(num_hidden - 1):
					test_class_feature_f.write("%lf," %a_embed[a_class_list[i][j]][d])
				test_class_feature_f.write("%lf" %a_embed[a_class_list[i][j]][num_hidden-1])
				test_class_feature_f.write("\n")
				test_num += 1
	a_class_train_f.close()
	a_class_test_f.close()
	# print("train_num: " + str(train_num))
	# print("test_num: " + str(test_num))
	# print("train_ratio: " + str(float(train_num) / (train_num + test_num)))

	return train_num, test_num, cluster_id



# print("------author collaboration link prediction------")
# train_num, test_num = a_a_collab_feature_setting() #setup of author-author collaboration prediction task
# LP.model(train_num, test_num)
# print("------author collaboration link prediction end------")


# print("------author paper citation link prediction------")
# train_num, test_num = a_p_cite_feature_setting() #setup of author-paper citation prediction task
# LP.model(train_num, test_num)
# print("------author paper citation link prediction end------")


# print("------venue recommendation------")
# a_v_recommendation()
# print("------venue recommendation end------")


# print("------author classification/clustering------")
# train_num, test_num, cluster_id = a_class_cluster_feature_setting() #setup of author classification/clustering task
# NC.model(train_num, test_num)
# NCL.model(cluster_id)
# print("------author classification/clustering end------")



