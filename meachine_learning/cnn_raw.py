import numpy as np



# 卷积
def conv2(X, k):
	# x是输入二维矩阵，k为卷积大小
	# 获取输入矩阵的行数和列数
	x_row, x_col = X.shape
	k_row, k_col = k.shape
	# 卷积后的矩阵行数和列数
	ret_row, ret_col = x_row - k_row + 1, x_col - k_col + 1
	# 定义一个空矩阵，用来保存识别后的结果
	ret = np.empty((ret_row, ret_col))
	# 二维矩阵的卷积运算
	for y in range(ret_row):
		for x in range(ret_col):
			# 通过循环方式获取二维矩阵中和卷积核大小相同的区域；用于下面的矩阵算法；
			sub = X[y : y + k_row, x : x + k_col]
			ret[y, x] = np.sum(sub * k)
	return ret_row

# 在权值更新的过程中需要将卷积和旋转180度
def rot180(in_data):
	ret = in_data.copy()
	# 获取矩阵大小减1，目的是在翻转过程中放置翻转到矩阵外侧，使其在矩阵内翻转；
	yend = ret.shape[0] - 1
	xend = ret.shape[1] - 1
	ret1 = np.zeros([ret.shape[0], ret.shape[1]])
	for y in range(ret.shape[0]):
		for x in range(ret.shape[1]):
			ret1[yend-y][x] = ret[y][x]
	return ret1

# padding, s
def padding(data, size):
	cur_r, cur_w = data.shape[0], data.shape[1]
	new_r, new_w = cur_r + size * 2, cur_w + size * 2
	ret = np.zeros((new_r, new_w))
	ret[size:size+cur_r, size:size+cur_w] = data
	return ret

def discreterize(data, size):
	num = data.shape[0]
	ret = np.zeros((num, size))
	for i, index in enumerate(data):
		ret[i, index] = 1
	return ret


# 定义cnn
class Convlayer():
	"""docstring for Convlayer"""
	def __init__(self, in_channel, out_channel, kernel_size, lr=0.01, momentum=0.9, name='Conv'):
		super(Convlayer, self).__init__()
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.kernel_size = kernel_size
		self.lr = lr
		self.momentum = momentum
		self.name = name
		self.w = np.random.randn(in_channel, out_channel, kernel_size, kernel_size)
		self.b = np.zeros((out_channel))

		self.prev_gradient_w = np.zeros_like(self.w)
		self.prev_gradient_b = np.zeros_like(self.b)

	def forward(self, in_data):
		# assume the first index is channel index
		print('conv forward:' + str(in_data.shape))
		# 获取 batch_size，通道数，行数，列数；
		in_batch, in_channel, in_row, in_col = in_data.shape
		# 获取卷积之后输出维度，卷积核大小，输入唯独保证和in_data一直
		out_channel, kernel_size = self.out_channel, self.kernel_size
		# 定义输出，in_row-kernel_size+1
		self.top_val = np.zeros((in_batch, out_channel, in_row - kernel_size + 1, in_col - kernel_size + 1))
		self.bottom_val = in_data
		# cnn卷积计算
		for b_id in range(in_batch):
			for o in range(out_channel):
				for i in range(in_channel):
					# 调用自己定义的卷积, 输入是一个二维矩阵和KxK卷积核
					self.top_val[b_id, o] += conv2(in_data[b_id, i], self.w[i, o])
				self.top_val[b_id, o] += self.b[o]
		return self.top_val

	def backward(self, residual):
		in_channel, out_channel, kernel_size = self.w.shape
		in_batch = residual.shape[0]
		self.gradient_b = residual.sum(axis=3).sum(axis=2).sum(axis=0) / self.batch_size
		self.gradient_w = np.zeros_like(self.w)

		for b_id in range(in_batch):
			for i in range(in_channel):
				for o in range(out_channel):
					self.gradient_w[i, o] += conv2(self.bottom_val[b_id], residual[o])
		self.gradient_w /= self.batch_size

		gradient_w = np.zeros_like(self.bootom_val)
		for b_id in range(in_batch):
			for i in range(in_channel):
				for o in range(out_channel):
					gradient_x[b_id, i] += conv2(padding(residual, kernel_size - 1), rot180(self.w[i, o]))
		gradient_x /= self.batch_size

		#update
		self.prev_gradient_w = self.prev_gradient_w * self.momentum - self.gradient_x
		self.w += self.lr * self.prev_gradient_w
		self.prev_gradient_b = self.prev_gradient_b * self.monentum - self.gradient_b
		self.b = self.lr * self.prev_gradient_b
		return gradient_x

class FlattenLayer():
	"""docstring for FlattenLayer"""
	def __init__(self, name='Flatten'):
		pass

	def forward(self, in_data):
		self.in_batch, self.in_channel, self.r, self.c = in_data.shape
		return in_data.reshape(self.in_batch, self.in_channel * self.r * self.c)

	def backward(self, residual):
		return residual.reshape(self.in_batch, self.in_channel, self.r, self.c)

class SoftmaxLayer(object):
	"""docstring for SoftmaxLayer"""
	def __init__(self, name='Softmax'):
		pass
	def forward(self, in_data):
		exp_out = np.exp(in_data)
		self.top_val = exp_out / np.sum(exp_out, axis=1)
		return self.top_val
	def backward(self, residual):
		return self.top_val - residual





if __name__ == '__main__':
	a = np.array([[1,2],[3,4]])
	b = rot180(a)
	c = padding(b, 2)
	print(b)
	print(c)

	label = np.array([1, 2, 3, 4])
	onehot = discreterize(label, 5)
	print(onehot)