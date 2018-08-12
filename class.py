class homeobj():
	"""build my homeobj"""
	def __init__(self, name, age, value):
		super(homeobj, self).__init__()
		self.name = name
		self.age = age
		self.value = value


	def showinfo(self):
		print('name: ', self.name, '\nage: ', self.age,'\nvalue: ', self.value)





liudi = homeobj('liudi', 25, 100)
liudi.showinfo()