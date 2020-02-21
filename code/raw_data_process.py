import string
import re
import numpy
import cPickle as pkl
import six.moves.cPickle as pickle
from collections import OrderedDict
import glob
import os
import sys
import random
import argparse
from subprocess import Popen, PIPE

tokenizer_cmd = ['./tokenizer.pl', '-l', 'en', '-q', '-']


def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type = str, default = '../data/academic_test/',
				   help='path to data')
	parser.add_argument('--A_n', type = int, default = 28646,
				   help = 'number of author node')
	parser.add_argument('--P_n', type = int, default = 21044,
				   help = 'number of paper node')
	parser.add_argument('--V_n', type = int, default = 18,
				   help = 'number of venue node')
	parser.add_argument('--T_split', type = int, default = 2012,
				   help = 'split time of train/test data')

	args = parser.parse_args()
	print(args)

	return args

args = read_args()


class paper:
	def __init__(self, title, author_list, time, venue, index, reference, abstract):
		self.title = title
		self.author_list = author_list
		self.time = time
		self.venue = venue
		self.index = index
		self.reference = reference
		self.abstract = abstract
	def __cmp__(self,other):
		return cmp(self.index, other.index)  


def read_paper_file():
	A_max = args.A_n
	P_max = args.P_n
	V_max = args.V_n
	T_split = args.T_split

	p_time = [0] * P_max
	p_venue = [0] * P_max
	p_list = []

	raw_p_f = open(args.data_path + "/academic_small.txt", "r")
	title_s = ''
	author_s = ''
	year_s = ''
	venue_s = ''
	index_s = ''
	ref_s = ''
	abstract_s = ''
	temp_s = ''
	for line in raw_p_f:
		line = line.strip()
		if line[0:2] == "#*":
			title_s = line[2:]
		elif line[0:2] == "#@":
			author_s = line[2:-1]
		elif line[0:2] == "#t":
			year_s = int(line[2:])
		elif line[0:2] == "#c":
			venue_s = line[2:]
		elif line[0:6] == "#index":
			index_s = int(line[6:])
		elif line[0:2] == "#%":
			ref_s = line[2:-1]
		elif line[0:2] == "#!":
			abstract_s = line[2:]
		elif line.strip() == '':
			if author_s != " ":
				temp = paper(title_s, author_s, year_s, venue_s, index_s, ref_s, abstract_s)
				p_list.append(temp)
				p_time[int(index_s)] = int(year_s)
				p_venue[int(index_s)] = int(venue_s)
				title_s = ''
				author_s = ''
				year_s = ''
				venue_s = ''
				index_s = ''
				ref_s = ''
				abstract_s = '' 
				temp_str = ''
	raw_p_f.close() 
	p_list = sorted(p_list)

	p_time_f = open(args.data_path + "p_time.txt", "w")
	for i in range(P_max):
		p_time_f.write("%d\t%d\n"%(i, p_time[i] - 2005))
		#p_time_f.write("%d\t%d\n"%(i, p_time[i] - 1995))
	p_time_f.close()


	p_a_list_train = [[] for k in range(P_max)]
	p_a_list_test = [[] for k in range(P_max)]
	a_p_list_train = [[] for k in range(A_max)]
	a_p_list_test = [[] for k in range(A_max)]
	p_p_cite_list_train = [[] for k in range(P_max)]
	p_p_cite_list_test = [[] for k in range(P_max)]
	v_p_list_train = [[] for k in range(V_max)]

	#test_count = 0 
	for i in range(len(p_list)):
		if p_time[p_list[i].index] < T_split:
			author_s = re.split(':', p_list[i].author_list)
			v_p_list_train[int(p_list[i].venue)].append(p_list[i].index)
			for j in range(len(author_s)):
				p_a_list_train[p_list[i].index].append(int(author_s[j]))
				a_p_list_train[int(author_s[j])].append(p_list[i].index)
			if len(p_list[i].reference):
				ref_s = re.split(':', p_list[i].reference);
				for k in range(len(ref_s)):
					if p_time[int(ref_s[k])] <= p_time[p_list[i].index]:#in case original data error
						p_p_cite_list_train[p_list[i].index].append(int(ref_s[k]))
		else:
			author_s = re.split(':', p_list[i].author_list)
			for j in range(len(author_s)):
				p_a_list_test[p_list[i].index].append(int(author_s[j]))
				a_p_list_test[int(author_s[j])].append(p_list[i].index)
			if len(p_list[i].reference) > 0:
				ref_s = re.split(':', p_list[i].reference);
				for k in range(len(ref_s)):
					#test_count += 1
					#if p_time[int(ref_s[k])] <= p_time[p_list[i].index]:
					p_p_cite_list_test[p_list[i].index].append(int(ref_s[k]))

	#print ("test_count: " + str(test_count))

	p_a_list_train_f = open(args.data_path + "p_a_list_train.txt", "w")
	p_a_list_test_f = open(args.data_path + "p_a_list_test.txt", "w")
	p_p_cite_list_train_f = open(args.data_path + "p_p_cite_list_train.txt", "w")
	p_p_cite_list_test_f = open(args.data_path + "p_p_cite_list_test.txt", "w")
	p_v_f = open(args.data_path + "p_v.txt", "w")

	for t in range(P_max):
		p_v_f.write(str(t) + "," + p_list[t].venue + "\n")
		if len(p_a_list_train[t]):
			p_a_list_train_f.write(str(t) + ":")
			for tt in range(len(p_a_list_train[t]) - 1):
				p_a_list_train_f.write(str(p_a_list_train[t][tt]) + ",")
			p_a_list_train_f.write(str(p_a_list_train[t][-1]))
			p_a_list_train_f.write("\n")

		if len(p_a_list_test[t]):
			p_a_list_test_f.write(str(t) + ":")
			for tt in range(len(p_a_list_test[t]) - 1):
				p_a_list_test_f.write(str(p_a_list_test[t][tt]) + ",")
			p_a_list_test_f.write(str(p_a_list_test[t][-1]))
			p_a_list_test_f.write("\n")
		 
		if len(p_p_cite_list_train[t]):
			p_p_cite_list_train_f.write(str(t)+":")
			for tt in range(len(p_p_cite_list_train[t])-1):
				p_p_cite_list_train_f.write(str(p_p_cite_list_train[t][tt])+",")
			p_p_cite_list_train_f.write(str(p_p_cite_list_train[t][-1]))
			p_p_cite_list_train_f.write("\n")

		if len(p_p_cite_list_test[t]):
			p_p_cite_list_test_f.write(str(t)+":")
			for tt in range(len(p_p_cite_list_test[t])-1):
				p_p_cite_list_test_f.write(str(p_p_cite_list_test[t][tt])+",")
			p_p_cite_list_test_f.write(str(p_p_cite_list_test[t][-1]))
			p_p_cite_list_test_f.write("\n")

	p_v_f.close()
	p_a_list_train_f.close()
	p_a_list_test_f.close()
	p_p_cite_list_train_f.close()
	p_p_cite_list_test_f.close()

	v_p_list_train_f = open(args.data_path + "v_p_list_train.txt", "w")
	for t in range(V_max):
		if len(v_p_list_train[t]):
			v_p_list_train_f.write(str(t)+":")
			for tt in range(len(v_p_list_train[t]) - 1):
				v_p_list_train_f.write(str(v_p_list_train[t][tt]) + ",")
			v_p_list_train_f.write(str(v_p_list_train[t][-1]))
			v_p_list_train_f.write("\n")
	v_p_list_train_f.close()

	a_p_list_train_f = open(args.data_path + "a_p_list_train.txt", "w")
	a_p_list_test_f = open(args.data_path + "a_p_list_test.txt", "w")

	for t in range(A_max):
		if len(a_p_list_train[t]):
			a_p_list_train_f.write(str(t) + ":")
			for tt in range(len(a_p_list_train[t]) - 1):
				a_p_list_train_f.write(str(a_p_list_train[t][tt]) + ",")
			a_p_list_train_f.write(str(a_p_list_train[t][-1]))
			a_p_list_train_f.write("\n")

		if len(a_p_list_test[t]):
			a_p_list_test_f.write(str(t) + ":")
			for tt in range(len(a_p_list_test[t]) - 1):
				a_p_list_test_f.write(str(a_p_list_test[t][tt]) + ",")
			a_p_list_test_f.write(str(a_p_list_test[t][-1]))
			a_p_list_test_f.write("\n") 
	
	a_p_list_train_f.close()
	a_p_list_test_f.close()

	return p_list
	

# p_list = []

# p_list = read_paper_file()






