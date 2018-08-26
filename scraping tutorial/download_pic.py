# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		简单练习之--下载国家地理美图 P9
'''
# -----------------------------------------------------------------------------------------------------
from bs4 import BeautifulSoup
import requests

URL = 'http://www.nationalgeographic.com.cn/animals/'

# find list of image holder
html = requests.get(URL).text
soup = BeautifulSoup(html, 'lxml')
img_ul = soup.find_all('ul', {'class': 'img_list'})
print(len(img_ul))

# creat a folder for these pictures
import os
os.makedirs('./img/', exist_ok=True)


# -----------------------------------------------------------------------------------------------------
'''
&usage:	download, find all picture urls and download them
		这一段稍微有点不是很理解，如何将网址数据下载到本地的呢？
'''
# -----------------------------------------------------------------------------------------------------
for ul in img_ul:
	imgs = ul.find_all('img')
	for img in imgs:
		url = img['src']
		r = requests.get(url, stream=True)
		imgae_name = url.split('/')[-1]
		with open('./img/%s' % imgae_name, 'wb') as f:
			for chunk in r.iter_content(chunk_size=128):
				f.write(chunk)
		print('saved %s' % imgae_name)

