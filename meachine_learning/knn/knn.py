#encoding:utf-8
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import operator
 
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
 
def createDataSet():
	train_data = []
	for i in range(1,32):
		pic = 'data/' + str(i) + '.png'
		lena = mpimg.imread(pic) # 读取和代码处于同一目录下的 lena.png
		gray = rgb2gray(lena)
		print(gray.shape, i)
		# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
		train_data.append(gray)

	train_data = np.array(train_data)
	print(train_data.shape)
	train_data = train_data.reshape(-1, 3600)
	print(train_data.shape)
	labels = ['横','竖','撇','点','横折','捺','提','横折勾','竖勾','横撇','横勾',
			  '竖弯钩','撇折','竖提','竖折','撇点','竖折折勾','斜勾','横撇弯钩','横折提',
			  '弯钩','横折弯钩','竖弯','横折弯','横折折折勾','横斜勾','横折折撇',
			  '竖折撇','竖折折','横折折','横折折折']
	return train_data, labels


def classify0(inX,dataSet,labels,k):
    #返回“数组”的行数，如果shape[1]返回的则是数组的列数
    dataSetSize = dataSet.shape[0]
    print(dataSetSize)
    #两个“数组”相减，得到新的数组
    diffMat = np.tile(inX,(dataSetSize,1))- dataSet
    #求平方
    sqDiffMat = diffMat **2
    #求和，返回的是一维数组
    sqDistances = sqDiffMat.sum(axis=1)
    #开方，即测试点到其余各个点的距离
    distances = sqDistances **0.5
    #排序，返回值是原数组从小到大排序的下标值
    sortedDistIndicies = distances.argsort()
    #定义一个空的字典
    classCount = {}
    for i in range(k):
        #返回距离最近的k个点所对应的标签值
        voteIlabel = labels[sortedDistIndicies[i]]
        #存放到字典中
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #排序 classCount.items() 输出键值对 key代表排序的关键字 True代表降序
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    #返回距离最小的点对应的标签
    return sortedClassCount[0][0]



dataSet,labels = createDataSet()
while True:
	filepath = input('请输入图片路径：')
	data = mpimg.imread(filepath)
	gray = rgb2gray(data)
	test_data = gray.reshape(-1,3600)
	result = classify0(test_data, dataSet, labels, 1)
	print(result)
