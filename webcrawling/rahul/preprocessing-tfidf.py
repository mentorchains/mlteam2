import csv
import math
import pandas
from collections import defaultdict

#Contains [Groups & Events], [General Discussion], [FAQ], [Deals], [Announcements], OTHER
generaldict = defaultdict(int)
generaldictidf = defaultdict(int)
generaltokens = []
generalnum = 0

#Contains [Community Created Device Types], [Community Created SmartApps]
communitycats = ["Community Created Device Types", "Community Created SmartApps"]
communitydict = defaultdict(int)
communitydictidf = defaultdict(int)
communitytokens = []
communitynum = 0

#Contains [Projects & Stories], [Apps & Clients], [Writing SmartApps], [SmartApps & Automations]
projectcats = ["Projects & Stories", "Apps & Clients", "Writing SmartApps", "SmartApps & Automations"]
projectdict = defaultdict(int)
projectdictidf = defaultdict(int)
projecttokens = []
projectnum = 0

#Contains [Devices & Integrations], [Hub Firmware Beta], [Connected Things], [Writing Device Types]
devicecats = ["Devices & Integrations", "Hub Firmware Beta", "Connected Things", "Writing Device Types"]
devicedict = defaultdict(int)
devicedictidf = defaultdict(int)
devicetokens = []
devicenum = 0


with open('web_scraping_data.csv', encoding="utf8") as data:
    reader = csv.reader(data)
    rownum = -1
    for row in reader:
        if(rownum != -1):
            title = row[1]
            category = row[2]
            content = row[5]
            contentwords = content.split(" ")
            replies = row[7]
            replywords = replies.split(" ")

            #Community Articles
            if(category in communitycats):
                for word in contentwords:
                    communitynum += 1
                    communitydict[word] += 1

                for word in replywords:
                    communitynum += 1
                    communitydict[word] += 1

            #Project and App Articles
            elif(category in projectcats):
                for word in contentwords:
                    projectnum += 1
                    projectdict[word] += 1

                for word in replywords:
                    projectnum += 1
                    projectdict[word] += 1

            #Device Articles
            elif(category in devicecats):
                for word in contentwords:
                    devicenum += 1
                    devicedict[word] += 1

                for word in replywords:
                    devicenum += 1
                    devicedict[word] += 1

            #General and Other
            else:
                for word in contentwords:
                    generalnum += 1
                    generaldict[word] += 1

                for word in replywords:
                    generalnum += 1
                    generaldict[word] += 1

        #Stop at nth row
        #if(rownum == 300):
        #    break

        rownum += 1

#Normalize term frequency by number of words
#Get a list of all the tokens
for key in generaldict:
    generaldict[key] /= float(generalnum)
    generaltokens.append(key)

for key in communitydict:
    communitydict[key] /= float(communitynum)
    communitytokens.append(key)

for key in projectdict:
    projectdict[key] /= float(projectnum)
    projecttokens.append(key)

for key in devicedict:
    devicedict[key] /= float(devicenum)
    devicetokens.append(key)

#Print dictionary in order of term frequency
#for a, b in sorted(generaldict.items(), key = lambda item: item[1], reverse=False):
#   print(a, ":", str(round(b*100, 2)) + "%")

#Inverse Document Frequency
#word: The term to find IDF for
#categories: All the document categories to compare
#Returns idfvar, the inverse document frequency of the word
def idf(word, categories):
    numcats = float(len(categories))
    termappear = 0
    idfvar = 0

    for cat in categories:
        if(word in cat):
            termappear += 1

    idfvar = 1.0 + math.log(numcats / termappear)

    return idfvar

generalremove = []
communityremove = []
projectremove = []
deviceremove = []

#Find IDFs for each category
for word in generaltokens:
    generaldictidf[word] = idf(word, (generaltokens, communitytokens, projecttokens, devicetokens))
    #Remove any words that appear in all categories (not unique)
    if(generaldictidf[word] == float(1)):
        generalremove.append(word)

for word in generalremove:
    generaltokens.remove(word)
    generaldict.pop(word)
    generaldictidf.pop(word)


for word in communitytokens:
    communitydictidf[word] = idf(word, (generaltokens, communitytokens, projecttokens, devicetokens))
    if(communitydictidf[word] == float(1)):
        communityremove.append(word)

for word in communityremove:
    communitytokens.remove(word)
    communitydict.pop(word)
    communitydictidf.pop(word)

for word in projecttokens:
    projectdictidf[word] = idf(word, (generaltokens, communitytokens, projecttokens, devicetokens))
    if(projectdictidf[word] == float(1)):
        projectremove.append(word)

for word in projectremove:
    projecttokens.remove(word)
    projectdict.pop(word)
    projectdictidf.pop(word)

for word in devicetokens:
    devicedictidf[word] = idf(word, (generaltokens, communitytokens, projecttokens, devicetokens))
    if(devicedictidf[word] == float(1)):
        deviceremove.append(word)

for word in deviceremove:
    devicetokens.remove(word)
    devicedict.pop(word)
    devicedictidf.pop(word)

generaltf = []
generalidf = []
communitytf = []
communityidf = []
projecttf = []
projectidf = []
devicetf = []
deviceidf = []

pandas.set_option('display.max_rows', None)

#Data Frames
for key in generaltokens:
    generaltf.append(generaldict[key])
    generalidf.append(generaldictidf[key])

generaldf = pandas.DataFrame()
generaldf["Word"] = generaltokens
generaldf["Term Frequency"] = generaltf
generaldf["Inverse Document Frequency"] = generalidf

print("GENERAL TABLE")
print(generaldf.sort_values(by=['Inverse Document Frequency']))

for key in communitytokens:
    communitytf.append(communitydict[key])
    communityidf.append(communitydictidf[key])

communitydf = pandas.DataFrame()
communitydf["Word"] = communitytokens
communitydf["Term Frequency"] = communitytf
communitydf["Inverse Document Frequency"] = communityidf

print("\n\nCOMMUNITY TABLE")
print(communitydf.sort_values(by=['Inverse Document Frequency']))

for key in projecttokens:
    projecttf.append(projectdict[key])
    projectidf.append(projectdictidf[key])

projectdf = pandas.DataFrame()
projectdf["Word"] = projecttokens
projectdf["Term Frequency"] = projecttf
projectdf["Inverse Document Frequency"] = projectidf

print("\n\nPROJECT TABLE")
print(projectdf.sort_values(by=['Inverse Document Frequency']))

for key in devicetokens:
    devicetf.append(devicedict[key])
    deviceidf.append(devicedictidf[key])

devicedf = pandas.DataFrame()
devicedf["Word"] = devicetokens
devicedf["Term Frequency"] = devicetf
devicedf["Inverse Document Frequency"] = deviceidf

print("\n\nDEVICE TABLE")
print(devicedf.sort_values(by=['Inverse Document Frequency']))