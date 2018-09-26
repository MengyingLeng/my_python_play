#! /usr/bin/env python
#! -*- coding=utf-8 -*-
 
#模拟两个正态分布的均值估计
 
from numpy import *
import numpy as np
import random
import copy


SIGMA = 6
EPS = 0.0001

#生成方差相同,均值不同的样本
def generate_data():	
	Miu1 = 20
	Miu2 = 40
	N = 1000
	X = mat(zeros((N,1)))
	for i in range(N):
		temp = random.uniform(0,1)
		if(temp > 0.3):
			# 均值为24.5，取值范围为23-26
			X[i] = temp + Miu1
		else:
			# 均值为41.5，取值范围为41-43
			X[i] = temp + Miu2
	return X
 
# EM算法
def my_EM(X):
	# 该模型包含两个单高斯
	k = 2
	# 数据量为N
	N = len(X)
	# 随机生成一个2x1的矩阵
	Miu = np.random.randn(2,1)
	# 注意要采用浮点数，给一个较合适的初始值较好
	Sigma = np.array([[15.],[2.]])
	# 每个分量的权值
	weight = np.array([[0.5],[0.5]])
	# 初始化1000个数据的后验概率1000x2
	Posterior = mat(zeros((N,2)))

	dominator = 0
	numerator = 0
	# 先求后验概率
	for iter in range(1000):
		for i in range(N):
			dominator = 0
			# estimate
			for j in range(k):
				# 求样本在当前模型下的整体概率
				dominator = dominator + weight[j] / (Sigma[j]) * np.exp(-1.0/(2.0*(Sigma[j])**2) * (X[i] - Miu[j])**2)
			# 求样本在某个高斯分量下的概率值，以及与整体概率的
			for j in range(k):
				numerator = weight[j] / (Sigma[j]) * np.exp(-1.0/(2.0*(Sigma[j])**2) * (X[i] - Miu[j])**2)
				Posterior[i,j] = numerator/dominator
		# 参数值放到旧的参数中			
		oldMiu = copy.deepcopy(Miu)
		# 得到后验概率
		print(Posterior)
		#最大化	
		for j in range(k):
			numerator = 0
			dominator = 0
			for i in range(N):
				numerator = numerator + Posterior[i,j] * X[i]
				dominator = dominator + Posterior[i,j]
			weight[j] = 1 / N * dominator
			Miu[j] = numerator/dominator

			numerator = 0
			for i in range(N):
				numerator = numerator + Posterior[i,j] * (( X[i] - Miu[j]) ** 2)
			Sigma[j] = np.sqrt(numerator/dominator)



		print ((abs(Miu - oldMiu)).sum()) 
			#print '\n'
		if (abs(Miu - oldMiu)).sum() < EPS:
			print(Miu,Sigma,weight,iter)
			break
 

if __name__ == '__main__':
	X = generate_data()
	print(X)
	my_EM(X)
