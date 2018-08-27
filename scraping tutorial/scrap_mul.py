# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		简单的Python scrap，多线程编程
'''
# -----------------------------------------------------------------------------------------------------

import multiprocessing as mp
import time
import re
from urllib.request import urlopen, urljoin
from bs4 import BeautifulSoup

base_url = 'https://morvanzhou.github.io/'

restricted_crawl = True

def crawl(url):
	response = urlopen(url)
	time.sleep(0.1)
	return response.read().decode('utf-8')


def parse(html):
	soup = BeautifulSoup(html, 'lxml')
	urls = soup.find_all('a', {'href': re.compile('^/.+?/$')})
	title = soup.find('h2').get_text().strip()
	page_urls = set([urljoin(base_url, url['href']) for url in urls])
	url = soup.find('meta', {'property': 'og:url'})['content']
	return title, page_urls, url


unseen = set([base_url,])
seen = set()

count, t1 = 1, time.time()

while len(unseen) != 0:
	if restricted_crawl and len(seen) > 20:
		break

	print('\nDdistributed crawling ...')
	htmls = [crawl(url) for url in unseen]

	print('\nDdistributed Parsing ...')
	results = [parse(html) for html in htmls]

	print('\nAnalysing ...')
	seen.update(unseen)
	unseen.clear()

	for title, page_urls, url in results:
		print(count, title, url)
		count += 1
		unseen.update(page_urls - seen)
	print(unseen)
print('total time: %.1f s' % (time.time() - t1, ))