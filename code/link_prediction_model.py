import random
import string
import re
import numpy
from itertools import *
import sklearn
from sklearn import linear_model
import sklearn.metrics as Metric
import csv
import argparse

parser = argparse.ArgumentParser(description = 'link prediction task')
parser.add_argument('--A_n', type = int, default = 28646,
			   help = 'number of author node')
parser.add_argument('--P_n', type = int, default = 21044,
			   help = 'number of paper node')
parser.add_argument('--V_n', type = int, default = 18,
			   help = 'number of venue node')
parser.add_argument('--data_path', type = str, default = '../data/academic_test/',
				   help='path to data')
parser.add_argument('--embed_d', type = int, default = 128,
			   help = 'embedding dimension')

args = parser.parse_args()
print(args)


def load_data(data_file_name, n_features, n_samples):
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = numpy.empty((n_samples, n_features))
        for i, d in enumerate(data_file):
            data[i] = numpy.asarray(d[:], dtype=numpy.float)    
        f.close  

        return data


def model(train_num, test_num):
	#train_data_f = args.data_path + "a_a_list_train_feature.txt"
	train_data_f = args.data_path + "train_feature.txt"
	train_data = load_data(train_data_f, args.embed_d + 3, train_num)
	train_features = train_data.astype(numpy.float32)[:,3:-1]
	train_target=train_data.astype(numpy.float32)[:,2]

	#print(train_target[1])
   	learner=linear_model.LogisticRegression()
   	learner.fit(train_features, train_target)
   	train_features = None
	train_target = None

   	print("training finish!")

   	#test_data_f = args.data_path + "a_a_list_test_feature.txt"
   	test_data_f = args.data_path + "test_feature.txt"
   	test_data = load_data(test_data_f, args.embed_d + 3, test_num)
   	test_id = test_data.astype(numpy.int32)[:,0:2]
   	test_features = test_data.astype(numpy.float32)[:,3:-1]
	test_target = test_data.astype(numpy.float32)[:,2]
	test_predict = learner.predict(test_features)
	test_features = None

	print("test prediction finish!")

	output_f = open(args.data_path + "link_prediction.txt", "w")
	for i in range(len(test_predict)):
	    output_f.write('%d, %d, %lf\n'%(test_id[i][0], test_id[i][1], test_predict[i]));
	output_f.close();

	AUC_score = Metric.roc_auc_score(test_target, test_predict)
	print("AUC: " + str(AUC_score))

	total_count = 0
	correct_count = 0
	true_p_count = 0
	false_p_count = 0
	false_n_count = 0 

	for i in range(len(test_predict)):
		total_count += 1
		if (int(test_predict[i]) == int(test_target[i])):
			correct_count += 1
		if (int(test_predict[i]) == 1 and int(test_target[i]) == 1):
			true_p_count += 1
		if (int(test_predict[i]) == 1 and int(test_target[i]) == 0):
			false_p_count += 1
		if (int(test_predict[i]) == 0 and int(test_target[i]) == 1):
			false_n_count += 1

	#print("accuracy: " + str(float(correct_count) / total_count))
	precision = float(true_p_count) / (true_p_count + false_p_count)
	#print("precision: " + str(precision))
	recall = float(true_p_count) / (true_p_count + false_n_count)
	#print("recall: " + str(recall))
	F1 = float(2 * precision * recall) / (precision + recall)
	print("F1: " + str(F1))



