# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		简单的Python scrap，基础练习
'''
# -----------------------------------------------------------------------------------------------------
from urllib.request import urlopen
html = urlopen("https://morvanzhou.github.io/discuss/").read().decode('utf-8')
print(html)

# 利用正则表达式选取信息
import re
res = re.findall(r"<strong>(.+?)</strong>", html, flags=re.DOTALL)
print("\n加粗的字体是:", res)

content = re.findall(r"<p>(.+?)</p>", html, flags=re.DOTALL)
print("\n段落内容是:", content)

links = re.findall(r'href="(.*?)', html, flags=re.DOTALL)
print("\n所有的links是:", links)


# --------------------------------------------------------------------------------------------------------
'''
&usage:		beautiful soup, easier than regular
'''
# --------------------------------------------------------------------------------------------------------
from bs4 import BeautifulSoup

# 选择解析形式'lxml'
soup = BeautifulSoup(html, features='lxml')
# 返回为h1的title
print(soup.h1)
# 返回为p的title
print('\n', soup.p)
# 返回所有的链接
all_href = soup.find_all('a')

# 最好使用dict.get('find name','default')进行取值，否则没有key的话就会报错
'''
all_href = [l.get('href','none') for l in all_href]
print(all_href)
'''
for l in all_href:
	print(l.get('href','none'))

# l.get_text()只显示内容
all_reply_content_wrapper = soup.find_all('a',{'class':'clicker'})
for l in all_reply_content_wrapper:
	print(l.get_text())

# 多级查询，注意开始要用find，而不是findall
all_reply = soup.find('li',{'class':'dropbtn'})
all_reply_content_wrapper = all_reply.find_all('li')
for l in all_reply_content_wrapper:
	print(l.get_text())


# ----------------------------------------------------------------------------------------------------------
'''
&usage:	使用正则表达式对爬虫数据进行处理
'''
# ----------------------------------------------------------------------------------------------------------
import re
# 查找图片链接，注意正则表达式的使用方法
img_links = soup.find_all('img', {'src': re.compile('.*?\.jpg')})
for link in img_links:
	print(link.get('src','none'))
# 查找提到的github地址
course_links = soup.find_all('a', {'href': re.compile('https://github.*')})
for link in course_links:
	print(link.get('href','none'))