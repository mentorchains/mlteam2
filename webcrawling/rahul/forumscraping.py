import requests
import pandas
from bs4 import BeautifulSoup
from collections import defaultdict

topics = []
tags = []
href = []
authors = []
content = []

base = "https://community.smartthings.com"

page = "/latest?no_definitions=true&page="

#Go through two scrolls
for num in range(2):
    response = requests.get(base + page + str(num))

    soup = BeautifulSoup(response.content, 'html.parser')

    #print(soup)

    #Find all the tags starting with 'a'

    #For all the topic headings
    for link in soup.find_all('meta', {'itemprop': 'name'}):
        #Add the topic to the list
        topics.append(link['content'])

    #For all the categories that appear under the topics
    for link in soup.find_all('span', {'class': 'category-name'}):
        #Add the category to the list
        tags.append(link.text)

    #For all the URLs that each topic leads to
    for link in soup.find_all('meta', {'itemprop': 'url'}):
        #Add the URL to the list
        url = link['content'].split("/")
        #Single out the extension (for practice)
        href.append("/" + url[3] + "/" + url[4] + "/" + url[5])

    ### OLD CODE ###
    #lambda makes a mini function. In this case the mini function is
        # sort by index=1, aka the number of times the tag appears.
    #reverse=True sorts it into descending order.
        
    #for k, v in sorted(tags.items(), key = lambda item: item[1], reverse=True):
    #for index in range(len(topics)):
    #   print("TOPIC: " + topics[index])
    #   print("CATEGORY: " + tags[index] + "\n")

#response = requests.get(base + href[0])
#print(BeautifulSoup(response.content, 'html.parser'))
#print("\n\n\n")

#Go to each topic page
for extension in href:
    response = requests.get(base + extension)
    soup = BeautifulSoup(response.content, 'html.parser')

    #print(soup)

    #Find the place where the username is located
    link1 = soup.find('span', {'itemprop': 'author'})

    #Go into the <a> tag inside the <span> tag
    link2 = link1.find('a')
    #Get the author from the text in the tag
    authors.append(link2.text)

    #Find the article body
    body = soup.find('div', {'itemprop': 'articleBody'})
    text = ""
    #Concatenate all the paragraphs that make up the article content
    for message in body.find_all('p'):
        text += message.text + " "

    #Add the content to the list
    content.append(text)
    #print(text)
    #print("\n")

#Create a table (Pandas Data Frame)
table = pandas.DataFrame()
table["Topic"] = topics
table["Category"] = tags
table["Author"] = authors
table["Text"] = content 

print(table)