import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import csv

tags = defaultdict(int)
title = []
url = []
category = []
reply=[]
view=[]
number=0
pagenum=100
base = "https://community.smartthings.com/latest?no_definitions=true&page="

for i in range(pagenum+1):
        response=requests.get(base+str(i-1))
        soup=BeautifulSoup(response.content,'html.parser')
        for link in soup.find_all('span', {'class': 'category-name'}):
                tags[link.text]+=1
                category.append(link.text)
                number+=1

        for link in soup.find_all('a', {'class': 'title raw-link raw-topic-link'}):
                title.append(link.text)
                url.append(link.get("href"))

        for link in soup.find_all('span', {'class': 'posts'}):
                reply.append(link.text)


        for link in soup.find_all('span', {'class': 'views'}):
                view.append(link.text)




for k,v in sorted(tags.items(),key=lambda item:item[1], reverse=True):
        print(k,':',v)

# for i in range(1,number+1):
#     print(category[i-1]+'\t\t'+title[i-1]+'\t\t'+url[i-1]+'\t\t'+reply[i-1]+'\t\t'+view[i-1])

print(number)


with open('data.csv', 'w') as csvfile:
    fieldnames = ['No.', 'Category', 'Title', 'URL','Replies','Views']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(1, number+1): 
        writer.writerow({'No.': i, 'Category': category[i-1],'Title': title[i-1], 'URL': url[i-1],'Replies':reply[i-1],'Views':view[i-1]})
  
