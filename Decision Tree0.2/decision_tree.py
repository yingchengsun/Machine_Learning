#-*- coding:utf-8 -*-
#Author:Jorbe
#Date:2014.06.14 16:10

import sys
from math import log
import operator
from numpy import mean
from data import get_dataset
from scipy.stats.mstats_basic import threshold
import random

sys.setrecursionlimit(1000000)
Max_Depth=10
def get_labels(train_file):
	'''
	返回所有数据集labels(列表)
	'''
	labels = []
	for index,line in enumerate(open(train_file,'rU').readlines()):
		label = line.strip().split(',')[-1]
		labels.append(label)
	return labels

def format_data(dataset_file):
	'''
	返回dataset(列表集合)和features(列表)
	'''
	'''
	dataset = []
	for index,line in enumerate(open(dataset_file,'rU').readlines()):
		line = line.strip()
		fea_and_label = line.split(',')
		dataset.append([float(fea_and_label[i]) for i in range(len(fea_and_label)-1)]+[fea_and_label[len(fea_and_label)-1]])
	#features = [dataset[0][i] for i in range(len(dataset[0])-1)]
	#sepal length（花萼长度）、sepal width（花萼宽度）、petal length（花瓣长度）、petal width（花瓣宽度）
	
	features=[]
	for j in range(19):
		features.append(str(j))
	#features = ['sepal_length','sepal_width','petal_length','petal_width']
	'''
	schema, X, y = get_dataset(train_file,'.')
	#labels = get_labels(train_file)
	labels =y
	train_dataset=map(list, X)
	for i in range(len(train_dataset)):
		train_dataset[i].append(y[i])
	train_features=[]
	
	'''
	for feature_name in schema.feature_names:
		train_features.append(feature_name)
	print schema.feature_names
	print type(schema.feature_names)
	print schema.nominal_values
	print type(schema.nominal_values)
	is_nominal=[]
	for i in range(len(schema.feature_names)):
		is_nominal.append(schema.is_nominal(i))
	print is_nominal
	print type(is_nominal)
	'''	
	return train_dataset,schema

def split_dataset(dataset,feature_index,labels):
	'''
	按指定feature划分数据集，返回四个列表:
	@dataset_less:指定特征项的属性值＜=该特征项平均值的子数据集
	@dataset_greater:指定特征项的属性值＞该特征项平均值的子数据集
	@label_less:按指定特征项的属性值＜=该特征项平均值切割后子标签集
	@label_greater:按指定特征项的属性值＞该特征项平均值切割后子标签集
	'''
	dataset_less = []
	dataset_greater = []
	label_less = []
	label_greater = []
	datasets = []
	n_data=len(dataset[0])-1

	for data in dataset:
		datasets.append(data[0:n_data])
	mean_value = mean(datasets,axis = 0)[feature_index]   #数据集在该特征项的所有取值的平均值
	#mean_value=0.5
	for data in dataset:
    		if data[feature_index] > mean_value:
      			dataset_greater.append(data)
      			label_greater.append(data[-1])
    		else:
      			dataset_less.append(data)
      			label_less.append(data[-1])
  	return dataset_less,dataset_greater,label_less,label_greater

def cal_entropy(dataset):
	'''
	计算数据集的熵大小
	'''
	n = len(dataset)    
	label_count = {}
	for data in dataset:
		label = data[-1]
		if label_count.has_key(label):
			label_count[label] += 1
		else:
			label_count[label] = 1
	entropy = 0
	for label in label_count:
		prob = float(label_count[label])/n
		entropy -= prob*log(prob,2)
	#print 'entropy:',entropy
	return entropy

def cal_info_gain(dataset,feature_index,schema,base_entropy):
	'''
	计算指定特征对数据集的信息增益值
	g(D,F) = H(D)-H(D/F) = entropy(dataset) - sum{1,k}(len(sub_dataset)/len(dataset))*entropy(sub_dataset)
	@base_entropy = H(D)
	'''
	part_data = {}

	condition_entropy=0.0
	label=[]
	for d in dataset:
		label.append(d[-1])
            
	if schema.is_nominal(feature_index):
		for i,ex in enumerate(dataset): 
			part_data.setdefault(ex[feature_index],[]).append(ex)
		for index,data in part_data.iteritems():
			condition_entropy += float(len(data))/len(dataset)*cal_entropy(data) 
		'''
		for value in schema.nominal_values[feature_index]:
			for data in dataset:
				if data[feature_index] ==value:
					dataset_greater.append(data)
					labels_greater.append(data[-1])
				else:
					dataset_less.append(data) 
					labels_less.append(data[-1])
			
			condition_entropy_temporary = float(len(dataset_less))/len(dataset)*cal_entropy(dataset_less) + float(len(dataset_greater))/len(dataset)*cal_entropy(dataset_greater)
			if condition_entropy_temporary>condition_entropy:
				condition_entropy=condition_entropy_temporary
				'''
	else:
		dataset = sorted(dataset,key=lambda x:x[feature_index])
		n_data=len(dataset)
		threshold=dataset[0][feature_index]
		dataset_less=dataset
		dataset_greater=[]
		for i,(ex1,ex2) in enumerate(zip(dataset[:-1],dataset[1:])):
			if ex1[-1] == ex2[-1]: #not a threshold
				continue
			else:
				threshold=ex1[feature_index]
				dataset_less=dataset[0:i]
				dataset_greater=dataset[i:n_data]
			condition_entropy_temporary = float(len(dataset_less))/len(dataset)*cal_entropy(dataset_less) + float(len(dataset_greater))/len(dataset)*cal_entropy(dataset_greater)
			#print condition_entropy_temporary
			if condition_entropy_temporary>condition_entropy:
				condition_entropy=condition_entropy_temporary
		part_data={threshold:dataset_less,1:dataset_greater}
	'''	
	datasets = []
	n_data=len(dataset[0])-1
	for data in dataset:
		datasets.append(data[0:n_data])
	
	mean_value = mean(datasets,axis = 0)[feature_index]    #计算指定特征的所有数据集值的平均值
	
	dataset_less = []
	dataset_greater = []
	for data in dataset:
		if data[feature_index] > mean_value:
			dataset_greater.append(data)
		else:
			dataset_less.append(data)
	#条件熵 H(D/F)
	condition_entropy = float(len(dataset_less))/len(dataset)*cal_entropy(dataset_less) + float(len(dataset_greater))/len(dataset)*cal_entropy(dataset_greater)
	#print 'info_gain:',base_entropy - condition_entropy
	'''
	return base_entropy - condition_entropy, part_data

def cal_info_gain_ratio(dataset,feature_index,schema):
	'''
	计算信息增益比  gr(D,F) = g(D,F)/H(D)
	'''	
	base_entropy = cal_entropy(dataset)
	'''
	if base_entropy == 0:
		return 1
	'''
	info_gain,part_data = cal_info_gain(dataset,feature_index,schema,base_entropy)
	info_gain_ratio = info_gain/base_entropy
	return info_gain_ratio,part_data
	
def choose_best_fea_to_split(dataset,schema):
	'''
	根据每个特征的信息增益比大小，返回最佳划分数据集的特征索引
	'''
	#base_entropy = cal_entropy(dataset)
	split_fea_index = -1
	max_info_gain_ratio = 0.0
	for i in range(len(schema.feature_names)):
		#info_gain = cal_info_gain(dataset,i,base_entropy)
		#info_gain_ratio = info_gain/base_entropy
		info_gain_ratio,part_data = cal_info_gain_ratio(dataset,i,schema)
		if info_gain_ratio > max_info_gain_ratio:
			max_info_gain_ratio = info_gain_ratio
			split_fea_index = i
	return split_fea_index,part_data

def most_occur_label(labels):
	'''
	返回数据集中出现次数最多的label
	'''
	label_count = {}
  	for label in labels:
    		if label not in label_count.keys():
      			label_count[label] = 1
    		else:
      			label_count[label] += 1
  	sorted_label_count = sorted(label_count.iteritems(),key = operator.itemgetter(1),reverse = True)
  	return sorted_label_count[0][0]

def build_tree(dataset,schema,depth=0):
	'''
	创建决策树
	@dataset:训练数据集
	@labels:数据集中包含的所有label(可重复)
	@features:可进行划分的特征集
	'''
	labels=[]
	for data in dataset:
		labels.append(data[-1])
	#若数据集为空,返回NULL
	if len(labels) == 0:
		return 'NULL'
	#若数据集中只有一种label,返回该label
	labelset=set(labels)
	if len(labelset) == 1:
		return labels[0]
	#若没有可划分的特征集,则返回数据集中出现次数最多的label
	if len(schema.feature_names) == 0:
		return most_occur_label(labels)
	#若数据集趋于稳定，则返回数据集中出现次数最多的label
	if cal_entropy(dataset) == 0:
		return most_occur_label(labels)
	split_feature_index,part_data = choose_best_fea_to_split(dataset,schema)
	split_feature = schema.feature_names[split_feature_index]
	decision_tree = {split_feature:{}}
	
	#若划分特征的信息增益比小于阈值,则返回数据集中出现次数最多的label
	#print cal_info_gain_ratio(dataset,split_feature_index)
	#if cal_info_gain_ratio(dataset,split_feature_index) < 0.001:
	if depth>Max_Depth:
		return most_occur_label(labels)
	del(schema.feature_names[split_feature_index])
	#dataset_less,dataset_greater,labels_less,labels_greater = split_dataset(dataset,split_feature_index,labels)
	#decesion_tree[split_feature]['<='] = build_tree(dataset_less,labels_less,schema,depth+1)
	#decesion_tree[split_feature]['>'] = build_tree(dataset_greater,labels_greater,schema,depth+1)

	for value,sub_data in part_data.iteritems():
		print depth
        decision_tree[split_feature][value] = build_tree(sub_data,schema,depth+1)
                
	return decision_tree

def store_tree(decision_tree,filename):
	'''
	把决策树以二进制格式写入文件
	'''
	import pickle
	writer = open(filename,'w')
	pickle.dump(decision_tree,writer)
	writer.close()

def read_tree(filename):
	'''
	从文件中读取决策树，返回决策树
	'''
	import pickle
	reader = open(filename,'rU')
	return pickle.load(reader)

def classify(decision_tree,schema,test_data):
	'''
	对测试数据进行分类, decesion_tree : {'petal_length': {'<=': {'petal_width': {'<=': 'Iris-setosa', '>': {'sepal_width': {'<=': 'Iris-versicolor', '>': {'sepal_length': {'<=': 'Iris-setosa', '>': 'Iris-versicolor'}}}}}}, '>': 'Iris-virginica'}} 
	'''
	while (type(decision_tree)==dict):
		key=decision_tree.keys()[0]
		feature_index = schema.feature_names.index(key)
		if schema.is_nominal(feature_index):
			try:
				decision_tree=decision_tree[key][test_data[feature_index]]
			except KeyError: 
				rand_index=random.randint(0,len(decision_tree[key].keys())-1)
				decision_tree=decision_tree[key].keys()[rand_index]
		else:
			threshold=decision_tree[key].keys()[0]
			if test_data[feature_index]>=threshold:
				decision_tree=decision_tree[key][threshold]
			else:
				decision_tree=decision_tree[key][1]
	return decision_tree
	'''
	first_fea = decision_tree.keys()[0]
	fea_index = schema.feature_names.index(first_fea)
	print test_data[fea_index]
	if schema.is_nominal(fea_index):
		sub_tree = decision_tree[first_fea][test_data[fea_index]]
		if type(sub_tree) == dict:
			return classify(sub_tree,schema,test_data)
		else:
			return sub_tree
	else:
		if test_data[fea_index] <= decision_tree.values()[0]:
			sub_tree = decision_tree[first_fea][0]
			if type(sub_tree) == dict:
				return classify(sub_tree,schema,test_data)
			else:
				return sub_tree
		else:
			sub_tree = decision_tree[first_fea][1]
			if type(sub_tree) == dict:
				return classify(sub_tree,schema,test_data)
			else:
				return sub_tree
	'''
def get_means(train_dataset):
	'''
	获取训练数据集各个属性的数据平均值
	'''
	dataset = []
	n_dataset=len(train_dataset[0])-1
	for data in train_dataset:
		dataset.append(data[0:n_dataset])
	mean_values = mean(dataset,axis = 0)   #数据集在该特征项的所有取值的平均值
	#mean_values=1
	return mean_values

def run(train_file,test_file):
	'''
	主函数
	'''
	train_dataset,schema = format_data(train_file)
	
	decision_tree = build_tree(train_dataset,schema)
	print 'decision_tree :',decision_tree
	
	store_tree(decision_tree,'decesion_tree')
	mean_values = get_means(train_dataset)
	#test_dataset,test_features = format_data(test_file)
	test_dataset, schema2 = format_data(test_file)
	n = len(test_dataset)
	correct = 0	
	
	for test_data in test_dataset:
		label = classify(decision_tree,schema2,test_data)
		#print 'classify_label  correct_label:',label,test_data[-1]
		if label == test_data[-1]:
			correct += 1
	print "准确率: ",correct/float(n)
	
#############################################################
if __name__ == '__main__':
	'''
	if len(sys.argv) != 3:
		print "please use: python decision.py train_file test_file"
		sys.exit()
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	run(train_file,test_file)
	'''
	#train_file ='voting.mat'
	train_file ='volcanoes.mat'
	test_file = 'volcanoes.mat'

	
  
    #attr_set=[]
    #for i in range(len(schema.feature_names)):
    #   attr_set.append(schema.is_nominal(i))    
	run(train_file,test_file)
	'''

	labels = get_labels(train_file)

	myset = set(labels)
	if len(myset)==1:
		print 'one'
	train_dataset,train_features = format_data(train_file)
	print labels
	print train_features
	'''