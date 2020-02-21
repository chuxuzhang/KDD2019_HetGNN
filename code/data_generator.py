import six.moves.cPickle as pickle
import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *


class input_data(object):
	def __init__(self, args):
		self.args = args

		a_p_list_train = [[] for k in range(self.args.A_n)]
		p_a_list_train = [[] for k in range(self.args.P_n)]
		p_p_cite_list_train = [[] for k in range(self.args.P_n)]
		v_p_list_train = [[] for k in range(self.args.V_n)]

		relation_f = ["a_p_list_train.txt", "p_a_list_train.txt",\
		 "p_p_citation_list.txt", "v_p_list_train.txt"]

		#store academic relational data 
		for i in range(len(relation_f)):
			f_name = relation_f[i]
			neigh_f = open(self.args.data_path + f_name, "r")
			for line in neigh_f:
				line = line.strip()
				node_id = int(re.split(':', line)[0])
				neigh_list = re.split(':', line)[1]
				neigh_list_id = re.split(',', neigh_list)
				if f_name == 'a_p_list_train.txt':
					for j in range(len(neigh_list_id)):
						a_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
				elif f_name == 'p_a_list_train.txt':
					for j in range(len(neigh_list_id)):
						p_a_list_train[node_id].append('a'+str(neigh_list_id[j]))
				elif f_name == 'p_p_citation_list.txt':
					for j in range(len(neigh_list_id)):
						p_p_cite_list_train[node_id].append('p'+str(neigh_list_id[j]))
				else:
					for j in range(len(neigh_list_id)):
						v_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
			neigh_f.close()

		#store paper venue 
		p_v = [0] * self.args.P_n
		p_v_f = open(self.args.data_path + 'p_v.txt', "r")
		for line in p_v_f:
			line = line.strip()
			p_id = int(re.split(',',line)[0])
			v_id = int(re.split(',',line)[1])
			p_v[p_id] = v_id
		p_v_f.close()

		#paper neighbor: author + citation + venue
		p_neigh_list_train = [[] for k in range(self.args.P_n)]
		for i in range(self.args.P_n):
			p_neigh_list_train[i] += p_a_list_train[i]
			p_neigh_list_train[i] += p_p_cite_list_train[i] 
			p_neigh_list_train[i].append('v' + str(p_v[i]))
		#print p_neigh_list_train[11846]

		self.a_p_list_train =  a_p_list_train
		self.p_a_list_train = p_a_list_train
		self.p_p_cite_list_train = p_p_cite_list_train
		self.p_neigh_list_train = p_neigh_list_train
		self.v_p_list_train = v_p_list_train

		if self.args.train_test_label != 2:
			self.triple_sample_p = self.compute_sample_p()

			#store paper content pre-trained embedding
			p_abstract_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			p_a_e_f = open(self.args.data_path + "p_abstract_embed.txt", "r")
			for line in islice(p_a_e_f, 1, None):
				values = line.split()
				index = int(values[0])
				embeds = np.asarray(values[1:], dtype='float32')
				p_abstract_embed[index] = embeds
			p_a_e_f.close()

			p_title_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			p_t_e_f = open(self.args.data_path + "p_title_embed.txt", "r")
			for line in islice(p_t_e_f, 1, None):
				values = line.split()
				index = int(values[0])
				embeds = np.asarray(values[1:], dtype='float32')
				p_title_embed[index] = embeds
			p_t_e_f.close()

			self.p_abstract_embed = p_abstract_embed
			self.p_title_embed = p_title_embed

			#store pre-trained network/content embedding
			a_net_embed = np.zeros((self.args.A_n, self.args.in_f_d))
			p_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			v_net_embed = np.zeros((self.args.V_n, self.args.in_f_d)) 
			net_e_f = open(self.args.data_path + "node_net_embedding.txt", "r")
			for line in islice(net_e_f, 1, None):
				line = line.strip()
				index = re.split(' ', line)[0]
				if len(index) and (index[0] == 'a' or index[0] == 'v' or index[0] == 'p'):
					embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
					if index[0] == 'a':
						a_net_embed[int(index[1:])] = embeds
					elif index[0] == 'v':
						v_net_embed[int(index[1:])] = embeds
					else:
						p_net_embed[int(index[1:])] = embeds
			net_e_f.close()

			p_v_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			p_v = [0] * self.args.P_n
			p_v_f = open(self.args.data_path + "p_v.txt", "r")
			for line in p_v_f:
				line = line.strip()
				p_id = int(re.split(',', line)[0])
				v_id = int(re.split(',', line)[1])
				p_v[p_id] = v_id
				p_v_net_embed[p_id] = v_net_embed[v_id]
			p_v_f.close()

			p_a_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			for i in range(self.args.P_n):
				if len(p_a_list_train[i]):
					for j in range(len(p_a_list_train[i])):
						a_id = int(p_a_list_train[i][j][1:])
						p_a_net_embed[i] = np.add(p_a_net_embed[i], a_net_embed[a_id])
					p_a_net_embed[i] = p_a_net_embed[i] / len(p_a_list_train[i])

			p_ref_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
			for i in range(self.args.P_n):
				if len(p_p_cite_list_train[i]):
					for j in range(len(p_p_cite_list_train[i])):
						p_id = int(p_p_cite_list_train[i][j][1:])
						p_ref_net_embed[i] = np.add(p_ref_net_embed[i], p_net_embed[p_id])
					p_ref_net_embed[i] = p_ref_net_embed[i] / len(p_p_cite_list_train[i])
				else:
					p_ref_net_embed[i] = p_net_embed[i]

			#empirically use 3 paper embedding for author content embeding generation
			a_text_embed = np.zeros((self.args.A_n, self.args.in_f_d * 3))
			for i in range(self.args.A_n):
				if len(a_p_list_train[i]):
					feature_temp = []
					if len(a_p_list_train[i]) >= 3:
						#id_list_temp = random.sample(a_p_list_train[i], 5)
						for j in range(3):
							feature_temp.append(p_abstract_embed[int(a_p_list_train[i][j][1:])])
					else:
						for j in range(len(a_p_list_train[i])):
							feature_temp.append(p_abstract_embed[int(a_p_list_train[i][j][1:])])
						for k in range(len(a_p_list_train[i]), 3):
							feature_temp.append(p_abstract_embed[int(a_p_list_train[i][-1][1:])])

					feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
					a_text_embed[i] = feature_temp

			#empirically use 5 paper embedding for author content embeding generation
			v_text_embed = np.zeros((self.args.V_n, self.args.in_f_d * 5))
			for i in range(self.args.V_n):
				if len(v_p_list_train[i]):
					feature_temp = []
					if len(v_p_list_train[i]) >= 5:
						#id_list_temp = random.sample(a_p_list_train[i], 5)
						for j in range(5):
							feature_temp.append(p_abstract_embed[int(v_p_list_train[i][j][1:])])
					else:
						for j in range(len(v_p_list_train[i])):
							feature_temp.append(p_abstract_embed[int(v_p_list_train[i][j][1:])])
						for k in range(len(v_p_list_train[i]), 5):
							feature_temp.append(p_abstract_embed[int(v_p_list_train[i][-1][1:])])

					feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
					v_text_embed[i] = feature_temp

			self.p_v = p_v
			self.p_v_net_embed = p_v_net_embed
			self.p_a_net_embed = p_a_net_embed
			self.p_ref_net_embed = p_ref_net_embed
			self.p_net_embed = p_net_embed
			self.a_net_embed = a_net_embed
			self.a_text_embed = a_text_embed
			self.v_net_embed = v_net_embed
			self.v_text_embed = v_text_embed

			#store neighbor set from random walk sequence 
			a_neigh_list_train = [[[] for i in range(self.args.A_n)] for j in range(3)]
			p_neigh_list_train = [[[] for i in range(self.args.P_n)] for j in range(3)]
			v_neigh_list_train = [[[] for i in range(self.args.V_n)] for j in range(3)]

			het_neigh_train_f = open(self.args.data_path + "het_neigh_train.txt", "r")
			for line in het_neigh_train_f:
				line = line.strip()
				node_id = re.split(':', line)[0]
				neigh = re.split(':', line)[1]
				neigh_list = re.split(',', neigh)
				if node_id[0] == 'a' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							a_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'p':
							a_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'v':
							a_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif node_id[0] == 'p' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							p_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						if neigh_list[j][0] == 'p':
							p_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						if neigh_list[j][0] == 'v':
							p_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
				elif node_id[0] == 'v' and len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							v_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						if neigh_list[j][0] == 'p':
							v_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						if neigh_list[j][0] == 'v':
							v_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))	
			het_neigh_train_f.close()
			#print a_neigh_list_train[0][1]

			#store top neighbor set (based on frequency) from random walk sequence 
			a_neigh_list_train_top = [[[] for i in range(self.args.A_n)] for j in range(3)]
			p_neigh_list_train_top = [[[] for i in range(self.args.P_n)] for j in range(3)]
			v_neigh_list_train_top = [[[] for i in range(self.args.V_n)] for j in range(3)]
			top_k = [10, 10, 3] #fix each neighor type size 
			for i in range(self.args.A_n):
				for j in range(3):
					a_neigh_list_train_temp = Counter(a_neigh_list_train[j][i])
					top_list = a_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 0 or j == 1:
						neigh_size = 10
					else:
						neigh_size = 3
					for k in range(len(top_list)):
						a_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(a_neigh_list_train_top[j][i]) and len(a_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(a_neigh_list_train_top[j][i]), neigh_size):
							a_neigh_list_train_top[j][i].append(random.choice(a_neigh_list_train_top[j][i]))

			for i in range(self.args.P_n):
				for j in range(3):
					p_neigh_list_train_temp = Counter(p_neigh_list_train[j][i])
					top_list = p_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 0 or j == 1:
						neigh_size = 10
					else:
						neigh_size = 3
					for k in range(len(top_list)):
						p_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(p_neigh_list_train_top[j][i]) and len(p_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(p_neigh_list_train_top[j][i]), neigh_size):
							p_neigh_list_train_top[j][i].append(random.choice(p_neigh_list_train_top[j][i]))

			for i in range(self.args.V_n):
				for j in range(3):
					v_neigh_list_train_temp = Counter(v_neigh_list_train[j][i])
					top_list = v_neigh_list_train_temp.most_common(top_k[j])
					neigh_size = 0
					if j == 0 or j == 1:
						neigh_size = 10
					else:
						neigh_size = 3
					for k in range(len(top_list)):
						v_neigh_list_train_top[j][i].append(int(top_list[k][0]))
					if len(v_neigh_list_train_top[j][i]) and len(v_neigh_list_train_top[j][i]) < neigh_size:
						for l in range(len(v_neigh_list_train_top[j][i]), neigh_size):
							v_neigh_list_train_top[j][i].append(random.choice(v_neigh_list_train_top[j][i]))

			a_neigh_list_train[:] = []
			p_neigh_list_train[:] = []
			v_neigh_list_train[:] = []

			self.a_neigh_list_train = a_neigh_list_train_top
			self.p_neigh_list_train = p_neigh_list_train_top
			self.v_neigh_list_train = v_neigh_list_train_top

			#store ids of author/paper/venue used in training 
			train_id_list = [[] for i in range(3)]
			for i in range(3):
				if i == 0:
					for l in range(self.args.A_n):
						if len(a_neigh_list_train_top[i][l]):
							train_id_list[i].append(l)
					self.a_train_id_list = np.array(train_id_list[i])
				elif i == 1:
					for l in range(self.args.P_n):
						if len(p_neigh_list_train_top[i][l]):
							train_id_list[i].append(l)
					self.p_train_id_list = np.array(train_id_list[i])
				else:
					for l in range(self.args.V_n):
						if len(v_neigh_list_train_top[i][l]):
							train_id_list[i].append(l)
					self.v_train_id_list = np.array(train_id_list[i])
			#print (len(self.v_train_id_list))		


	def het_walk_restart(self):
		a_neigh_list_train = [[] for k in range(self.args.A_n)]
		p_neigh_list_train = [[] for k in range(self.args.P_n)]
		v_neigh_list_train = [[] for k in range(self.args.V_n)]

		#generate neighbor set via random walk with restart
		node_n = [self.args.A_n, self.args.P_n, self.args.V_n]
		for i in range(3):
			for j in range(node_n[i]):
				if i == 0:
					neigh_temp = self.a_p_list_train[j]
					neigh_train = a_neigh_list_train[j]
					curNode = "a" + str(j)
				elif i == 1:
					neigh_temp = self.p_a_list_train[j]
					neigh_train = p_neigh_list_train[j]
					curNode = "p" + str(j)
				else:
					neigh_temp = self.v_p_list_train[j]
					neigh_train = v_neigh_list_train[j]
					curNode = "v" + str(j)
				if len(neigh_temp):
					neigh_L = 0
					a_L = 0
					p_L = 0
					v_L = 0
					while neigh_L < 100: #maximum neighbor size = 100
						rand_p = random.random() #return p
						if rand_p > 0.5:
							if curNode[0] == "a":
								curNode = random.choice(self.a_p_list_train[int(curNode[1:])])
								if p_L < 46: #size constraint (make sure each type of neighobr is sampled)
									neigh_train.append(curNode)
									neigh_L += 1
									p_L += 1
							elif curNode[0] == "p":
								curNode = random.choice(self.p_neigh_list_train[int(curNode[1:])])
								if curNode != ('a' + str(j)) and curNode[0] == 'a' and a_L < 46:
									neigh_train.append(curNode)
									neigh_L += 1
									a_L += 1
								elif curNode[0] == 'v':
									if v_L < 11:
										neigh_train.append(curNode)
										neigh_L += 1
										v_L += 1
							elif curNode[0] == "v":
								curNode = random.choice(self.v_p_list_train[int(curNode[1:])])
								if p_L < 46:
									neigh_train.append(curNode)
									neigh_L +=1
									p_L += 1
						else:
							if i == 0:
								curNode = ('a' + str(j))
							elif i == 1:
								curNode = ('p' + str(j))
							else:
								curNode = ('v' + str(j))

		for i in range(3):
			for j in range(node_n[i]):
				if i == 0:
					a_neigh_list_train[j] = list(a_neigh_list_train[j])
				elif i == 1:
					p_neigh_list_train[j] = list(p_neigh_list_train[j])
				else:
					v_neigh_list_train[j] = list(v_neigh_list_train[j])

		neigh_f = open(self.args.data_path + "het_neigh_train.txt", "w")
		for i in range(3):
			for j in range(node_n[i]):
				if i == 0:
					neigh_train = a_neigh_list_train[j]
					curNode = "a" + str(j)
				elif i == 1:
					neigh_train = p_neigh_list_train[j]
					curNode = "p" + str(j)
				else:
					neigh_train = v_neigh_list_train[j]
					curNode = "v" + str(j)
				if len(neigh_train):
					neigh_f.write(curNode + ":")
					for k in range(len(neigh_train) - 1):
						neigh_f.write(neigh_train[k] + ",")
					neigh_f.write(neigh_train[-1] + "\n")
		neigh_f.close()


	def compute_sample_p(self):
		print("computing sampling ratio for each kind of triple ...")
		window = self.args.window
		walk_L = self.args.walk_L
		A_n = self.args.A_n
		P_n = self.args.P_n
		V_n = self.args.V_n

		total_triple_n = [0.0] * 9 # nine kinds of triples
		het_walk_f = open(self.args.data_path + "het_random_walk.txt", "r")
		centerNode = ''
		neighNode = ''

		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'a':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[0] += 1
								elif neighNode[0] == 'p':
									total_triple_n[1] += 1
								elif neighNode[0] == 'v':
									total_triple_n[2] += 1
					elif centerNode[0]=='p':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[3] += 1
								elif neighNode[0] == 'p':
									total_triple_n[4] += 1
								elif neighNode[0] == 'v':
									total_triple_n[5] += 1
					elif centerNode[0]=='v':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[6] += 1
								elif neighNode[0] == 'p':
									total_triple_n[7] += 1
								elif neighNode[0] == 'v':
									total_triple_n[8] += 1
		het_walk_f.close()

		for i in range(len(total_triple_n)):
			total_triple_n[i] = self.args.batch_s / (total_triple_n[i] * 10)
		print("sampling ratio computing finish.")

		return total_triple_n


	def sample_het_walk_triple(self):
		print ("sampling triple relations ...")
		triple_list = [[] for k in range(9)]
		window = self.args.window
		walk_L = self.args.walk_L
		A_n = self.args.A_n
		P_n = self.args.P_n
		V_n = self.args.V_n
		triple_sample_p = self.triple_sample_p # use sampling to avoid memory explosion

		het_walk_f = open(self.args.data_path + "het_random_walk.txt", "r")
		centerNode = ''
		neighNode = ''
		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'a':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[0]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									# random negative sampling get similar performance as noise distribution sampling
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[0].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[1]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[1].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[2]:
									negNode = random.randint(0, V_n - 1)
									while len(self.v_p_list_train[negNode]) == 0:
										negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[2].append(triple)
					elif centerNode[0]=='p':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[3]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[3].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[4]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[4].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[5]:
									negNode = random.randint(0, V_n - 1)
									while len(self.v_p_list_train[negNode]) == 0:
										negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[5].append(triple)
					elif centerNode[0]=='v':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[6]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[6].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[7]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[7].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[8]:
									negNode = random.randint(0, V_n - 1)
									while len(self.v_p_list_train[negNode]) == 0:
										negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[8].append(triple)
		het_walk_f.close()

		return triple_list




