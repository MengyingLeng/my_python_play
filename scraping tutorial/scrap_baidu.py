# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		简单练习之--爬取百度百科练习p6
'''
# -----------------------------------------------------------------------------------------------------
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import random

# “网页爬虫”在百度百科的地址为：(base_url+history[0])
base_url = "https://baike.baidu.com"
# 保存爬过的网页
history = ["/item/%E7%BD%91%E7%BB%9C%E7%88%AC%E8%99%AB/5162711"]
# 生成所需爬虫的值，history中的最后一个地址
url = base_url + history[-1]

html = urlopen(url).read().decode('utf-8')
soup = BeautifulSoup(html, features='lxml')
print(soup.find('h1').get_text(),'	url: ', history[-1])


# -----------------------------------------------------------------------------------------------------
'''
&usage:		find valid urls，随机找到一个出现的百度百科相关链接，使用正则表达式
'''
# -----------------------------------------------------------------------------------------------------
sub_urls = soup.find_all('a', {'target':'_blank', 'href': re.compile('/item/(%.{2})+$')})
if len(sub_urls) != 0:
	history.append(random.sample(sub_urls, 1)[0]['href'])
else:
	history.pop()
print(history)


# -----------------------------------------------------------------------------------------------------
'''
&usage:		find valid urls，随机找到一个出现的百度百科相关链接，使用正则表达式
'''
# -----------------------------------------------------------------------------------------------------
for i in range(20):
	url = base_url + history[-1]

	html = urlopen(url).read().decode('utf-8')
	soup = BeautifulSoup(html, features='lxml')
	print(i, soup.find('h1').get_text(),'	url: ', history[-1])
	sub_urls = soup.find_all('a', {'target':'_blank', 'href': re.compile('/item/(%.{2})+$')})
	if len(sub_urls) != 0:
		history.append(random.sample(sub_urls, 1)[0]['href'])
	else:
		history.pop()

print(history)