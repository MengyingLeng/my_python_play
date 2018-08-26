# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		简单练习之--request方法
&install:	pip install requests
'''
# -----------------------------------------------------------------------------------------------------
import requests
import webbrowser




# -----------------------------------------------------------------------------------------------------
'''
	get:	正常打开网页、不往服务器传数据
	error:	不知道为什么不能讲参数传给requests
'''
# -----------------------------------------------------------------------------------------------------
param = {'wb': 'python'}
r = requests.get('http://www.baidu.com/s', params=param)
print(r.url)
webbrowser.open(r.url)


# -----------------------------------------------------------------------------------------------------
'''
	post:	账号登录、搜索内容、上传图片、上传文件、往服务器传数据
	note:	requests.post(link,data)其中link是你将要去的链接，不是当前链接
	web:	http://pythonscraping.com/pages/files/processing.php
			http://pythonscraping.com/pages/files/processing2.php
			http://pythonscraping.com/pages/cookies/login.html
'''
# -----------------------------------------------------------------------------------------------------
# 输入数据
data = {'firstname':'hongwen', 'lastname':'sun'}
r = requests.post('http://pythonscraping.com/pages/files/processing.php', data)
print(r.text)
# 上传图片
file = {'uploadFile': open('E:\\z.jpg', 'rb')}
r = requests.post('http://pythonscraping.com/pages/files/processing2.php', files=file)
print(r.text)
# 登录用户
payload = {'username': 'hongwen', 'password': 'password'}
r = requests.post('http://pythonscraping.com/pages/cookies/welcome.php', data=payload)
print(r.cookies.get_dict())
r = requests.post('http://pythonscraping.com/pages/cookies/welcome.php', cookies=r.cookies)
print(r.text)


# -----------------------------------------------------------------------------------------------------
'''
	session:	使用会话来进行登录
'''
# -----------------------------------------------------------------------------------------------------
session = requests.Session()
payload = {'username': 'hongwen', 'password': 'password'}
r = session.post('http://pythonscraping.com/pages/cookies/welcome.php', data=payload)
print(r.cookies.get_dict())
r = session.get('http://pythonscraping.com/pages/cookies/welcome.php')
print(r.text)
