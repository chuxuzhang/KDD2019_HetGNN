import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from args import read_args
import numpy as np
import string
import re
import math
args = read_args()


class HetAgg(nn.Module):
	def __init__(self, args, feature_list, a_neigh_list_train, p_neigh_list_train, v_neigh_list_train,\
		 a_train_id_list, p_train_id_list, v_train_id_list):
		super(HetAgg, self).__init__()
		embed_d = args.embed_d
		in_f_d = args.in_f_d
		self.args = args 
		self.P_n = args.P_n
		self.A_n = args.A_n
		self.V_n = args.V_n
		self.feature_list = feature_list
		self.a_neigh_list_train = a_neigh_list_train
		self.p_neigh_list_train = p_neigh_list_train
		self.v_neigh_list_train = v_neigh_list_train
		self.a_train_id_list = a_train_id_list
		self.p_train_id_list = p_train_id_list
		self.v_train_id_list = v_train_id_list

		#self.fc_a_agg = nn.Linear(embed_d * 4, embed_d)
			
		self.a_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.p_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.v_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)

		self.a_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.p_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.v_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)

		self.a_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)
		self.p_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)
		self.v_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)

		self.softmax = nn.Softmax(dim = 1)
		self.act = nn.LeakyReLU()
		self.drop = nn.Dropout(p = 0.5)
		self.bn = nn.BatchNorm1d(embed_d)
		self.embed_d = embed_d


	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
				nn.init.xavier_normal_(m.weight.data)
				#nn.init.normal_(m.weight.data)
				m.bias.data.fill_(0.1)


	def a_content_agg(self, id_batch): #heterogeneous content aggregation
		embed_d = self.embed_d
		#print len(id_batch)
		# embed_d = in_f_d, it is flexible to add feature transformer (e.g., FC) here 
		#print (id_batch)
		a_net_embed_batch = self.feature_list[6][id_batch]

		a_text_embed_batch_1 = self.feature_list[7][id_batch, :embed_d][0]
		a_text_embed_batch_2 = self.feature_list[7][id_batch, embed_d : embed_d * 2][0]
		a_text_embed_batch_3 = self.feature_list[7][id_batch, embed_d * 2 : embed_d * 3][0]

		concate_embed = torch.cat((a_net_embed_batch, a_text_embed_batch_1, a_text_embed_batch_2,\
		 a_text_embed_batch_3), 1).view(len(id_batch[0]), 4, embed_d)

		concate_embed = torch.transpose(concate_embed, 0, 1)
		all_state, last_state = self.a_content_rnn(concate_embed)

		return torch.mean(all_state, 0)


	def p_content_agg(self, id_batch):
		embed_d = self.embed_d
		p_a_embed_batch = self.feature_list[0][id_batch]
		p_t_embed_batch = self.feature_list[1][id_batch]
		p_v_net_embed_batch = self.feature_list[2][id_batch]
		p_a_net_embed_batch = self.feature_list[3][id_batch]
		p_net_embed_batch = self.feature_list[5][id_batch]

		concate_embed = torch.cat((p_a_embed_batch, p_t_embed_batch, p_v_net_embed_batch,\
		 p_a_net_embed_batch, p_net_embed_batch), 1).view(len(id_batch[0]), 5, embed_d)

		concate_embed = torch.transpose(concate_embed, 0, 1)
		all_state, last_state = self.p_content_rnn(concate_embed)

		return torch.mean(all_state, 0)


	def v_content_agg(self, id_batch):
		embed_d = self.embed_d
		v_net_embed_batch = self.feature_list[8][id_batch]
		v_text_embed_batch_1 = self.feature_list[9][id_batch, :embed_d][0]
		v_text_embed_batch_2 = self.feature_list[9][id_batch, embed_d: 2 * embed_d][0]
		v_text_embed_batch_3 = self.feature_list[9][id_batch, 2 * embed_d: 3 * embed_d][0]
		v_text_embed_batch_4 = self.feature_list[9][id_batch, 3 * embed_d: 4 * embed_d][0]
		v_text_embed_batch_5 = self.feature_list[9][id_batch, 4 * embed_d:][0]

		concate_embed = torch.cat((v_net_embed_batch, v_text_embed_batch_1, v_text_embed_batch_2, v_text_embed_batch_3,\
			v_text_embed_batch_4, v_text_embed_batch_5), 1).view(len(id_batch[0]), 6, embed_d)

		concate_embed = torch.transpose(concate_embed, 0, 1)
		all_state, last_state = self.v_content_rnn(concate_embed)
		
		return torch.mean(all_state, 0)


	def node_neigh_agg(self, id_batch, node_type): #type based neighbor aggregation with rnn 
		embed_d = self.embed_d

		if node_type == 1 or node_type == 2:
			batch_s = int(len(id_batch[0]) / 10)
		else:
			#print (len(id_batch[0]))
			batch_s = int(len(id_batch[0]) / 3)

		if node_type == 1:
			neigh_agg = self.a_content_agg(id_batch).view(batch_s, 10, embed_d)
			neigh_agg = torch.transpose(neigh_agg, 0, 1)
			all_state, last_state  = self.a_neigh_rnn(neigh_agg)
		elif node_type == 2:
			neigh_agg = self.p_content_agg(id_batch).view(batch_s, 10, embed_d)
			neigh_agg = torch.transpose(neigh_agg, 0, 1)
			all_state, last_state  = self.p_neigh_rnn(neigh_agg)
		else:
			neigh_agg = self.v_content_agg(id_batch).view(batch_s, 3, embed_d)
			neigh_agg = torch.transpose(neigh_agg, 0, 1)
			all_state, last_state  = self.v_neigh_rnn(neigh_agg)
		neigh_agg = torch.mean(all_state, 0).view(batch_s, embed_d)
		
		return neigh_agg


	def node_het_agg(self, id_batch, node_type): #heterogeneous neighbor aggregation
		a_neigh_batch = [[0] * 10] * len(id_batch)
		p_neigh_batch = [[0] * 10] * len(id_batch)
		v_neigh_batch = [[0] * 3] * len(id_batch)
		for i in range(len(id_batch)):
			if node_type == 1:
				a_neigh_batch[i] = self.a_neigh_list_train[0][id_batch[i]]
				p_neigh_batch[i] = self.a_neigh_list_train[1][id_batch[i]]
				v_neigh_batch[i] = self.a_neigh_list_train[2][id_batch[i]]
			elif node_type == 2:
				a_neigh_batch[i] = self.p_neigh_list_train[0][id_batch[i]]
				p_neigh_batch[i] = self.p_neigh_list_train[1][id_batch[i]]
				v_neigh_batch[i] = self.p_neigh_list_train[2][id_batch[i]]
			else:
				a_neigh_batch[i] = self.v_neigh_list_train[0][id_batch[i]]
				p_neigh_batch[i] = self.v_neigh_list_train[1][id_batch[i]]
				v_neigh_batch[i] = self.v_neigh_list_train[2][id_batch[i]]

		a_neigh_batch = np.reshape(a_neigh_batch, (1, -1))
		a_agg_batch = self.node_neigh_agg(a_neigh_batch, 1)
		p_neigh_batch = np.reshape(p_neigh_batch, (1, -1))
		p_agg_batch = self.node_neigh_agg(p_neigh_batch, 2)
		v_neigh_batch = np.reshape(v_neigh_batch, (1, -1))
		v_agg_batch = self.node_neigh_agg(v_neigh_batch, 3)

		#attention module
		id_batch = np.reshape(id_batch, (1, -1))
		if node_type == 1:
			c_agg_batch = self.a_content_agg(id_batch)
		elif node_type == 2:
			c_agg_batch = self.p_content_agg(id_batch)
		else:
			c_agg_batch = self.v_content_agg(id_batch)

		c_agg_batch_2 = torch.cat((c_agg_batch, c_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
		a_agg_batch_2 = torch.cat((c_agg_batch, a_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
		p_agg_batch_2 = torch.cat((c_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
		v_agg_batch_2 = torch.cat((c_agg_batch, v_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)

		#compute weights
		concate_embed = torch.cat((c_agg_batch_2, a_agg_batch_2, p_agg_batch_2,\
		 v_agg_batch_2), 1).view(len(c_agg_batch), 4, self.embed_d * 2)
		if node_type == 1:
			atten_w = self.act(torch.bmm(concate_embed, self.a_neigh_att.unsqueeze(0).expand(len(c_agg_batch),\
			 *self.a_neigh_att.size())))
		elif node_type == 2:
			atten_w = self.act(torch.bmm(concate_embed, self.p_neigh_att.unsqueeze(0).expand(len(c_agg_batch),\
			 *self.p_neigh_att.size())))
		else:
			atten_w = self.act(torch.bmm(concate_embed, self.v_neigh_att.unsqueeze(0).expand(len(c_agg_batch),\
			 *self.v_neigh_att.size())))
		atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, 4)

		#weighted combination
		concate_embed = torch.cat((c_agg_batch, a_agg_batch, p_agg_batch,\
		 v_agg_batch), 1).view(len(c_agg_batch), 4, self.embed_d)
		weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d)

		return weight_agg_batch


	def het_agg(self, triple_index, c_id_batch, pos_id_batch, neg_id_batch):
		embed_d = self.embed_d
		# batch processing
		# nine cases for academic data (author, paper, venue)
		if triple_index == 0:
			c_agg = self.node_het_agg(c_id_batch, 1)
			p_agg = self.node_het_agg(pos_id_batch, 1)
			n_agg = self.node_het_agg(neg_id_batch, 1)
		elif triple_index == 1:
			c_agg = self.node_het_agg(c_id_batch, 1)
			p_agg = self.node_het_agg(pos_id_batch, 2)
			n_agg = self.node_het_agg(neg_id_batch, 2)
		elif triple_index == 2:
			c_agg = self.node_het_agg(c_id_batch, 1)
			p_agg = self.node_het_agg(pos_id_batch, 3)
			n_agg = self.node_het_agg(neg_id_batch, 3)
		elif triple_index == 3:
			c_agg = self.node_het_agg(c_id_batch, 2)
			p_agg = self.node_het_agg(pos_id_batch, 1)
			n_agg = self.node_het_agg(neg_id_batch, 1)
		elif triple_index == 4:
			c_agg = self.node_het_agg(c_id_batch, 2)
			p_agg = self.node_het_agg(pos_id_batch, 2)
			n_agg = self.node_het_agg(neg_id_batch, 2)	
		elif triple_index == 5:
			c_agg = self.node_het_agg(c_id_batch, 2)
			p_agg = self.node_het_agg(pos_id_batch, 3)
			n_agg = self.node_het_agg(neg_id_batch, 3)	
		elif triple_index == 6:
			c_agg = self.node_het_agg(c_id_batch, 3)
			p_agg = self.node_het_agg(pos_id_batch, 1)
			n_agg = self.node_het_agg(neg_id_batch, 1)		
		elif triple_index == 7:
			c_agg = self.node_het_agg(c_id_batch, 3)
			p_agg = self.node_het_agg(pos_id_batch, 2)
			n_agg = self.node_het_agg(neg_id_batch, 2)	
		elif triple_index == 8:
			c_agg = self.node_het_agg(c_id_batch, 3)
			p_agg = self.node_het_agg(pos_id_batch, 3)
			n_agg = self.node_het_agg(neg_id_batch, 3)
		elif triple_index == 9: #save learned node embedding
			embed_file = open(self.args.data_path + "node_embedding.txt", "w")
			save_batch_s = self.args.mini_batch_s
			for i in range(3):
				if i == 0:
					batch_number = int(len(self.a_train_id_list) / save_batch_s)
				elif i == 1:
					batch_number = int(len(self.p_train_id_list) / save_batch_s)
				else:
					batch_number = int(len(self.v_train_id_list) / save_batch_s)
				for j in range(batch_number):
					if i == 0:
						id_batch = self.a_train_id_list[j * save_batch_s : (j + 1) * save_batch_s]
						out_temp = self.node_het_agg(id_batch, 1) 
					elif i == 1:
						id_batch = self.p_train_id_list[j * save_batch_s : (j + 1) * save_batch_s]
						out_temp = self.node_het_agg(id_batch, 2)
					else:
						id_batch = self.v_train_id_list[j * save_batch_s : (j + 1) * save_batch_s]
						out_temp = self.node_het_agg(id_batch, 3)
					out_temp = out_temp.data.cpu().numpy()
					for k in range(len(id_batch)):
						index = id_batch[k]
						if i == 0:
							embed_file.write('a' + str(index) + " ")
						elif i == 1:
							embed_file.write('p' + str(index) + " ")
						else:
							embed_file.write('v' + str(index) + " ")
						for l in range(embed_d - 1):
							embed_file.write(str(out_temp[k][l]) + " ")
						embed_file.write(str(out_temp[k][-1]) + "\n")

				if i == 0:
					id_batch = self.a_train_id_list[batch_number * save_batch_s : -1]
					out_temp = self.node_het_agg(id_batch, 1) 
				elif i == 1:
					id_batch = self.p_train_id_list[batch_number * save_batch_s : -1]
					out_temp = self.node_het_agg(id_batch, 2) 
				else:
					id_batch = self.v_train_id_list[batch_number * save_batch_s : -1]
					out_temp = self.node_het_agg(id_batch, 3) 
				out_temp = out_temp.data.cpu().numpy()
				for k in range(len(id_batch)):
					index = id_batch[k]
					if i == 0:
						embed_file.write('a' + str(index) + " ")
					elif i == 1:
						embed_file.write('p' + str(index) + " ")
					else:
						embed_file.write('v' + str(index) + " ")
					for l in range(embed_d - 1):
						embed_file.write(str(out_temp[k][l]) + " ")
					embed_file.write(str(out_temp[k][-1]) + "\n")
			embed_file.close()
			return [], [], []

		return c_agg, p_agg, n_agg


	def aggregate_all(self, triple_list_batch, triple_index):
		c_id_batch = [x[0] for x in triple_list_batch]
		pos_id_batch = [x[1] for x in triple_list_batch]
		neg_id_batch = [x[2] for x in triple_list_batch]

		c_agg, pos_agg, neg_agg = self.het_agg(triple_index, c_id_batch, pos_id_batch, neg_id_batch)

		return c_agg, pos_agg, neg_agg


	def forward(self, triple_list_batch, triple_index):
		c_out, p_out, n_out = self.aggregate_all(triple_list_batch, triple_index)
		return c_out, p_out, n_out


def cross_entropy_loss(c_embed_batch, pos_embed_batch, neg_embed_batch, embed_d):
	batch_size = c_embed_batch.shape[0] * c_embed_batch.shape[1]
	
	c_embed = c_embed_batch.view(batch_size, 1, embed_d)
	pos_embed = pos_embed_batch.view(batch_size, embed_d, 1)
	neg_embed = neg_embed_batch.view(batch_size, embed_d, 1)

	out_p = torch.bmm(c_embed, pos_embed)
	out_n = - torch.bmm(c_embed, neg_embed)

	sum_p = F.logsigmoid(out_p)
	sum_n = F.logsigmoid(out_n)
	loss_sum = - (sum_p + sum_n)

	#loss_sum = loss_sum.sum() / batch_size

	return loss_sum.mean()

