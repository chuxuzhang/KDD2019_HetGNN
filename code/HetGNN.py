import torch
import torch.optim as optim
import data_generator
import tools
from args import read_args
from torch.autograd import Variable
import numpy as np
import random
torch.set_num_threads(2)
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


class model_class(object):
	def __init__(self, args):
		super(model_class, self).__init__()
		self.args = args
		self.gpu = args.cuda

		input_data = data_generator.input_data(args = self.args)
		#input_data.gen_het_rand_walk()

		self.input_data = input_data

		if self.args.train_test_label == 2: #generate neighbor set of each node
			input_data.het_walk_restart()
			print ("neighbor set generation finish")
			exit(0)

		feature_list = [input_data.p_abstract_embed, input_data.p_title_embed,\
		input_data.p_v_net_embed, input_data.p_a_net_embed, input_data.p_ref_net_embed,\
		input_data.p_net_embed, input_data.a_net_embed, input_data.a_text_embed,\
		input_data.v_net_embed, input_data.v_text_embed]

		for i in range(len(feature_list)):
			feature_list[i] = torch.from_numpy(np.array(feature_list[i])).float()

		if self.gpu:
			for i in range(len(feature_list)):
				feature_list[i] = feature_list[i].cuda()
		#self.feature_list = feature_list

		a_neigh_list_train = input_data.a_neigh_list_train
		p_neigh_list_train = input_data.p_neigh_list_train
		v_neigh_list_train = input_data.v_neigh_list_train

		a_train_id_list = input_data.a_train_id_list
		p_train_id_list = input_data.p_train_id_list
		v_train_id_list = input_data.v_train_id_list

		self.model = tools.HetAgg(args, feature_list, a_neigh_list_train, p_neigh_list_train, v_neigh_list_train,\
		 a_train_id_list, p_train_id_list, v_train_id_list)

		if self.gpu:
			self.model.cuda()
		self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
		self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay = 0)
		self.model.init_weights()


	def model_train(self):
		print ('model training ...')
		if self.args.checkpoint != '':
			self.model.load_state_dict(torch.load(self.args.checkpoint))
		
		self.model.train()
		mini_batch_s = self.args.mini_batch_s
		embed_d = self.args.embed_d

		for iter_i in range(self.args.train_iter_n):
			print ('iteration ' + str(iter_i) + ' ...')
			triple_list = self.input_data.sample_het_walk_triple()
			min_len = 1e10
			for ii in range(len(triple_list)):
				if len(triple_list[ii]) < min_len:
					min_len = len(triple_list[ii])
			batch_n = int(min_len / mini_batch_s)
			print (batch_n)
			for k in range(batch_n):
				c_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
				p_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
				n_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])

				for triple_index in range(len(triple_list)):
					triple_list_temp = triple_list[triple_index]
					triple_list_batch = triple_list_temp[k * mini_batch_s : (k + 1) * mini_batch_s]
					c_out_temp, p_out_temp, n_out_temp = self.model(triple_list_batch, triple_index)

					c_out[triple_index] = c_out_temp
					p_out[triple_index] = p_out_temp
					n_out[triple_index] = n_out_temp

				loss = tools.cross_entropy_loss(c_out, p_out, n_out, embed_d)

				self.optim.zero_grad()
				loss.backward()
				self.optim.step() 

				if k % 100 == 0:
					print ("loss: " + str(loss))

			if iter_i % self.args.save_model_freq == 0:
				torch.save(self.model.state_dict(), self.args.model_path + "HetGNN_" + str(iter_i) + ".pt")
				# save embeddings for evaluation
				triple_index = 9 
				a_out, p_out, v_out = self.model([], triple_index)
			print ('iteration ' + str(iter_i) + ' finish.')



if __name__ == '__main__':
	args = read_args()
	print("------arguments-------")
	for k, v in vars(args).items():
		print(k + ': ' + str(v))

	#fix random seed
	random.seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed_all(args.random_seed)

	#model 
	model_object = model_class(args)

	if args.train_test_label == 0:
		model_object.model_train()

