<1> Introduction 

code of HetGNN in KDD2019 paper: Heterogeneous Graph Neural Network 
Contact: Chuxu Zhang (czhang11@nd.edu)

<2> How to use

python HetGNN.py [parameters]
(enable GPU: python HetGNN.py --cuda 1)

#test academic data: (author) A_n - 28646, (paper) P_n - 21044, ((venue) V_n - 18

<3> Data requirement
a_p_list_train.txt: paper neighbor list of each author in training data
p_a_list_train.txt: author neighbor list of each paper in training data
p_p_citation_list.txt: paper citation neighbor list of each paper 
v_p_list_train.txt: paper neighbor list of each venue in training data
p_v.txt: venue of each paper
p_title_embed.txt: pre-trained paper title embedding
p_abstract_embed.txt: pre-trained paper abstract embedding
node_net_embedding.txt: pre-trained node embedding by network embedding
het_neigh_train.txt: generated neighbor set of each node by random walk with re-start 
het_random_walk.txt: generated random walks as node sequences (corpus) for model training

<4> Model evaluation and raw data processing code will be uploaded late

<5> If you find code useful, please consider citing our work.

inproceedings{Zhang:2019:HGN:3292500.3330961,
 author = {Zhang, Chuxu and Song, Dongjin and Huang, Chao and Swami, Ananthram and Chawla, Nitesh V.},
 
 title = {Heterogeneous Graph Neural Network},
 
 booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
 
 series = {KDD '19},
 
 year = {2019},
 
 isbn = {978-1-4503-6201-6},
 
 location = {Anchorage, AK, USA},
 
 pages = {793--803},
 
 numpages = {11},
 
 url = {http://doi.acm.org/10.1145/3292500.3330961},
 
 doi = {10.1145/3292500.3330961},
 
 acmid = {3330961},
 
 publisher = {ACM},
 
 address = {New York, NY, USA},
 
 keywords = {graph embedding, graph neural networks, heterogeneous graphs},
}

