import gzip
import re
import urllib.request
import bs4
from collections import deque
import os

def download_url(url):
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; rv:37.0) Gecko/20100101 Firefox/37.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
        }
    req = urllib.request.Request(url, headers = header)
    con = urllib.request.urlopen(req, timeout = 100)
    doc = con.read()
    con.close()
    data = gzip.decompress(doc)
    data = data.decode('ascii', errors = 'ignore')
    return data

# delete blank and \n
def delete_blank(data):
    pat = re.compile('\n')
    data_s = pat.sub('', data)
    return data_s

# find all urls
def extract_urls(data):
    pat = re.compile('href=\"(.+?)\"')
    urls = pat.findall(data)
    return urls

# find all odds urls
def extract_even_urls(data):
    i = 1
    while i<len(data):
        data.pop(i)
        i = i+1
    return data

queue1 = deque()
queue2 = deque()
queue3 = deque()
queue4 = deque()
queue5 = deque()
queue6 = deque()

# first layer
url = 'http://jobs.monster.com/browse'
data = download_url(url)
data_n = delete_blank(data)

pat_11 = re.compile('<h2 class="fnt5">Browse Jobs By Category</h2>(.+?)Browse more<span class="sr-only">Web Jobs</span></a></div>')
str_url_11 = pat_11.findall(data_n)[0]

# Catch the top-level categroy labels
urls_1 = extract_urls(str_url_11)
urls_1 = extract_even_urls(urls_1)
pat_12 = re.compile('class=\"fnt4\" title=\"(.+?)\"')
titles_1 = pat_12.findall(str_url_11)


for i in range(len(urls_1)):
    try:
        data = download_url(urls_1[i])
        print("extract ------->"+" Big Catergory "+  titles_1[i])
    except:
        print('wrong!!!!!!' + '---->' + titles_1[i])
        queue1.append(urls_1[i])
        continue
    
    #Second Layer
    directory = 'F:/'+ 'STA242PROJECT8/'+titles_1[i] +'/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Catch the second level category labels 
    pg = 99
    for k in range(pg):
        urls_3 = urls_1[i] + '?page=' + str(k + 1)
        try:
            data = download_url(urls_3)
            print("extract----->"+ titles_1[i]+"||"+' page '+ str(k+1))
        except:
            print("wrong!!!"+titles_1[i]+"||"+' page '+ str(k+1))
            queue2.append(urls_3)
            continue
        #
        linkre = re.compile('href=\"(.+?jobview.+?|.+?job-openings.+?)\"')
        webls = linkre.findall(data)
        pattern = re.compile('&#39;')
        webls = [pattern.sub('\'', x) for x in webls]
        linkre1 = re.compile('datetime=\"(.+?)\"')
        datetime = linkre1.findall(data)
        linkre2 = re.compile('itemprop=\"datePosted\">(.+?)</time>\r')
        itemprop = linkre2.findall(data)
        linkre3 = re.compile('<span itemprop=\"title\">(.+?)</span></a>')        
        title_job = linkre3.findall(data)
        #Catch jobs in each page
        for m in range(len(webls)):    
            urlnew = webls[m]
            try:
                data = download_url(urlnew)
            except:
                print('wrong1!'+ titles_1[i]+"||"+' page '+ str(k+1)+' Job' + str(m+1))
                queue3.append(urlnew)
                save_path = directory + 'page'+str(k+1)+'_'+str(m+1)+'.txt'
                f = open(save_path, 'a', encoding = 'utf-8')
                f.write(title_job[m]+ '\n' + datetime[m] +' '+ itemprop[m])
                f.close()
                continue
            #
            try:
                soup =bs4.BeautifulSoup(data)
                jobname = soup.title.string
            except:
                print('wrong4!'+ titles_1[i]+"||"+' page '+ str(k+1)+' Job' + str(m+1))
                queue6.append(urlnew)
                save_path = directory + 'page'+str(k+1)+'_'+str(m+1)+'.txt'
                f = open(save_path, 'a', encoding = 'utf-8')
                f.write(title_job[m]+ '\n' + datetime[m] +' '+ itemprop[m])
                f.close()
                continue
            #
            try:
                jobbody = soup.find(id = 'TrackingJobBody')
                jobtext = jobbody.getText()
                content = jobtext + '\n' + datetime[m] +' '+ itemprop[m]
                save_path = directory + 'page'+str(k+1)+'_'+str(m+1)+'.txt'
                f = open(save_path, 'a', encoding = 'utf-8')
                f.write(jobname +'\n' + content)
                f.close()
                print('extract----->'+ titles_1[i]+"||"+' page '+ str(k+1)+' Job' + str(m+1))
            except:
                try:      
                    data_n = delete_blank(data)
                    pat = re.compile('<div class=\"jobview-section\">(.+?)</div><!-- ./jobview-section -->')
                    content = pat.findall(data_n)[0]+ '\n' + datetime[m] +' '+ itemprop[m]
                    save_path = directory + 'page'+str(k+1)+'_'+str(m+1)+'.txt'
                    f = open(save_path, 'a', encoding = 'utf-8')
                    f.write(jobname +'\n' + content)
                    f.close()
                    print('extract----->'+ titles_1[i]+"||"+' page '+ str(k+1)+' Job' + str(m+1))
                except:
                    try:
                        save_path = directory + 'page'+str(k+1)+'_'+str(m+1)+'.txt'
                        f = open(save_path, 'a', encoding = 'utf-8')
                        f.write(jobname+ '\n' + datetime[m] +' '+ itemprop[m])
                        f.close()
                        print('wrong2! '+ titles_1[i]+'||'+' page '+ str(k+1)+' Job' + str(m+1))
                        queue4.append(urlnew)
                    except:
                        save_path = directory + 'page'+str(k+1)+'_'+str(m+1)+'.txt'
                        f = open(save_path, 'a', encoding = 'utf-8')
                        f.write(title_job[m]+ '\n' + datetime[m] +' '+ itemprop[m])
                        f.close()
                        print('wrong3! '+ titles_1[i]+"||"+' page '+ str(k+1)+' Job' + str(m+1))
                        queue5.append(urlnew)
                        continue
       


"""
                  except:
                        print("warning"+ titles_1[i]+"||"+titles_2[j]+' page '+ str(k+1)+' Job' + str(m+1)
                        queue4.append(urlnew)
                        save_path = directory + 'page'+str(k+1)+'_'+str(m+1)+'.txt'
                        f = open(save_path, 'a', encoding = 'utf-8')
                        f.write('\n' + jobname)
                        continue
                        linkre3 = re.compile('href=\".+?jobview.+?|.+?job-openings.+?\".+?title=\"(.+?)\">
                             try:
                    save_path = directory + 'page'+str(k+1)+'_'+str(m+1)+'.txt'
                    f = open(save_path, 'a', encoding = 'utf-8')
                    f.write('\n'+jobname +'\n' + content)
                    f.close()
                except:
"""
