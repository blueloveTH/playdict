from urllib3 import HTTPConnectionPool
 
pool = HTTPConnectionPool('cn.bing.com', maxsize=1)
url = 'http://cn.bing.com/dict/clientsearch?mkt=zh-CN&setLang=zh&q='

def fetch_html(q):
    r = pool.request('GET', url+q, redirect=False)
    return r.data