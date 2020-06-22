#!/usr/bin/env python
# coding: utf-8

# In[99]:


from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq
url = 'https://community.smartthings.com/latest'


# In[100]:


import pandas as pd


# In[101]:


import requests


# In[102]:


import csv

filename = "Task1+.csv"
#f = open(filename, "w")
#headers = "Topic, Category, Tag, URL, Replies, Views"
#f.write(headers)


# In[103]:


df = pd.DataFrame()


# In[104]:


uClient = uReq(url)


# In[105]:


page_html = uClient.read()


# In[106]:


uClient.close()


# In[107]:


page_soup = soup(page_html, "html.parser")


# In[108]:


print(page_soup)


# In[109]:


def page_search(num_page):
    base = 'https://community.smartthings.com/latest?no_definitions=true&page='
    r = requests.get(base + str(num_page)).text
    bs = soup(r, "html.parser")
    
    containers = page_soup.findAll("tr", {"class": "topic-list-item"})
    topics = []
    categories = []
    tags = []
    urls = []
    for i in containers:
        topic = i.td.a.text    
        category = i.td.div.find('span', {'class' : 'category-name'}).text      
        tag = i.td.div.div.text.replace('\n', '').strip()
        url = i.td.a.get('href')
        topics.append(topic)
        categories.append(category)
        tags.append(tag)
        urls.append(url)
        
   
    
    all_replies = page_soup.findAll("td", {'class': 'replies'})
    replies = []
    for r in all_replies:
        reply = r.text.replace('\n','').strip()
        replies.append(reply)

        
    views = []
    all_views = page_soup.findAll("td", {'class': 'views'})
    for v in all_views:
        view = v.text.replace('\n','').strip()
        views.append(view)
        
    
    dataset = pd.DataFrame({'topic': topics, 'category': categories, 'tag': tags, 'url': urls, 'reply': replies, 'views': views})
    #dataset.to_csv(filename, index = True, header = True)
    return dataset


# In[110]:


df = []
for page in range(10):
    d = page_search(page)
    #d.to_csv(filename, mode='a', header=True)
    df.append(d)
df = pd.concat(df).reset_index()
df.align(df, join = 'left')
df.to_csv(filename, mode= 'w', header=True)



df


# In[111]:


#define the lists containing everything we need


# In[ ]:





# In[ ]:




