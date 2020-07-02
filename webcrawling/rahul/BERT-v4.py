from bs4 import BeautifulSoup
from collections import defaultdict
import csv
import math
import matplotlib
import numpy
import inspect
from keras.preprocessing.sequence import pad_sequences #Requires TensorFlow installation
import pandas
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel, AdamW, get_linear_schedule_with_warmup
import random
import re
import requests
import seaborn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#V4: Categories are from community.smartthings front page
#V4: Input data are ENTIRE articles

#Walkthrough Link: https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03

articledict = defaultdict(int)
bertTokens = []
bertIds = []
tags = []
numtags = []
articletags = []
num = 0

#Initiate the Bert Tokenizer
bertTokenize = BertTokenizer.from_pretrained('bert-base-uncased')

seed = 24
max_len = 256

url = []

base = "https://community.smartthings.com"
page = "/latest?no_definitions=true&page="

#Go through infinite scrolling
for num in range(5):
    response = requests.get(base + page + str(num))

    soup = BeautifulSoup(response.content, 'html.parser')

    #For all the categories that appear under the topics
    for link in soup.find_all('span', {'class': 'category-name'}):
        #Add the category to the list
        tags.append(link.text)

    #For all the URLs that each topic leads to
    for link in soup.find_all('meta', {'itemprop': 'url'}):
        #Add the URL to the list
        url.append(link['content'])


generalcats =  ["Announcements", "General Discussion", "Groups & Events", "Mobile Release Notes", "Platform Updates", 
                "Device Integrations", "Documentation Updates", "Hub Firmware Release Notes"] #GENERAL, 0
wikicats = ["Custom Solutions", "How-To", "FAQ", "Projects & Stores", "Ideas and Suggestions"] #WIKI, 1
devcats = ["Devices and Integrations", "Capability Type Suggestions", "Writing Device Types", "Device Ideas", 
            "Connected Things", "Deals", "Community Created Device Types"] #DEVICES & INTEGRATIONS, 2
appcats = ["Apps & Clients", "Mobile Tips & Tricks", "Features & Feedback", "iOS", "Android", "Windows", "SmartThings (Samsung Connect)"] #Apps & Clients, 3
autocats = ["SmartApps & Automations", "Writing SmartApps", "Community Created SmartApps", "Automation Ideas", "FAQ", "webCoRE", "Rules API"] #SmartApps & Automations, 4
procats = ["Developer Programs", "Hub Firmware Beta", "Tutorials", "Support"] #Developer Programs, 5

#Assign each category to a number
numcategories = 6
for element in tags:
    if(element in procats):
        numtags.append(5)
    elif(element in autocats):
        numtags.append(4)
    elif(element in appcats):
        numtags.append(3)
    elif(element in devcats):
        numtags.append(2)
    elif(element in wikicats):
        numtags.append(1)
    else:
        numtags.append(0)

print(numtags)

completed = 0
#Go to each topic page
for link in url:
    response = requests.get(link)
    soup = BeautifulSoup(response.content, 'html.parser')

    #Find the article body
    body = soup.find('div', {'itemprop': 'articleBody'})
    text = ""
    #Concatenate all the paragraphs that make up the article content
    for message in body.find_all('p'):
        text += message.text + " "

    #Shallow Tokenizer: Split the current article body into sentences
    #sentences = list(tokenizer.split_into_sentences(text))
   
    #Make article lowercase
    lowers = text.lower()
        
    #Remove punctuation
    #[]: Match single characters...  ^: Everything Except  #\w: Word characters  #\s: And space characters
    s = re.sub(r'[^\w\s]', '', lowers)

    #Remove pure numbers
    #\b: Start at the beginning of each word  #\d+: Capture all digits (greedy)  #\s: And make sure it ends with a space (the term is purely digits)
    s = re.sub(r'\b\d+\s', '', s)
    
    #Add necessary start/stop tokens for BERT
    sformatted = "[CLS] " + s + " [SEP]"

    #Use the Bert Tokenizer
    currenttokens = bertTokenize.tokenize(sformatted)

    #Cut length to max_len if necessary
    if(len(currenttokens) > max_len):
        currenttokens = currenttokens[0:max_len-1]

    bertTokens.append(currenttokens)

    #Add a label based on article category
    articletags.append(numtags[completed])

    #print(bertTokenize.tokenize(sformatted))

    #Go through n articles
    #if(completed == 10):
    #    break

    completed += 1
    print(str(completed) + "/" + str(len(url)) + " completed.")

#Create ID list for the tokens
for element in bertTokens:
    bertIds.append(bertTokenize.convert_tokens_to_ids(element))

#Pad the ID list so that all elements have a length of max_len
bertIds = pad_sequences(bertIds, maxlen=max_len, padding="post")

#Split data for training and testing (validation)
train_input_array, val_input_array, train_label, val_label = train_test_split(bertIds, articletags, test_size=0.15, random_state=seed)

#TrainTestSplit returns numpy arrays, so they must be converted to lists for BatchEncodePlus
train_input = train_input_array.tolist()
val_input = val_input_array.tolist()

#Special Tokens added (CLS/SEP), Want attention masks, Pad to max_len words, Already tokenized, Return pytorch tensors
train_encoded = bertTokenize.batch_encode_plus(train_input, add_special_tokens=True, return_attention_masks=True, pad_to_max_length=True, max_length=max_len, is_pretokenized=True, return_tensors='pt')

val_encoded = bertTokenize.batch_encode_plus(val_input, add_special_tokens=True, return_attention_masks=True, pad_to_max_length=True, max_length=max_len, is_pretokenized=True, return_tensors='pt')

#Convert data into tensors (required format)
##Sentences
train_input = train_encoded['input_ids']
val_input = val_encoded['input_ids']

##Labels
train_label_tensor = torch.Tensor(train_label).to(torch.int64)
val_label_tensor = torch.Tensor(val_label).to(torch.int64)

##Masks
train_masks = train_encoded['attention_mask']
val_masks = val_encoded['attention_mask']

training_data = TensorDataset(train_input, train_masks, train_label_tensor)
testing_data = TensorDataset(val_input, val_masks, val_label_tensor)

#Make a BERT model
##Not making another attention model, Not concerned with hidden states
bertModel = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=numcategories, output_attentions=False, output_hidden_states=False)

#Create Data Loaders
##Perform the training on the data
##Batch size indicates how many sentences are run at a time
batchsize = 32
train_dataloader = DataLoader(training_data, sampler=RandomSampler(training_data), batch_size=batchsize)
val_dataloader = DataLoader(testing_data, sampler=RandomSampler(testing_data), batch_size=batchsize)

#Set up optimization
##lr = learning rate, eps = epsilon values
##Learning rate describes how drastically the model will be updated with each step
optimizer = AdamW(bertModel.parameters(), lr = 1e-5, eps=1e-8)

##Epochs indicate how many times the the model will be updated (number of steps)
epochs = 10

##Specify how many steps the optimization should run
##Default number of warmup steps, training steps increases with size of training data
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*epochs)

random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cpu')
bertModel.to(device)
#print(device)

#Set the model's dictionary to the specified file
#bertModel.load_state_dict(torch.load('Model/finetuned_bert_epoch_1_gpu_trained.model', map_location=device))

#Evaluation
##Change to evaluation mode
bertModel.eval()

val_loss = 0
predictions = []
actual = []

for batch in val_dataloader:

    batch = tuple(element.to(device) for element in batch)

    inputs = {'input_ids':batch[0], 'attention_mask':batch[1], 'labels':batch[2]}

    #for x in range(5):
        #print(inputs['input_ids'][x], inputs['labels'][x], inputs['attention_mask'][x])

    with torch.no_grad():
        output = bertModel(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        #outputs = bertModel(inputs['input_ids'], inputs['attention_mask'], inputs['labels'])

    #print(output)
    #elementloss = output[0]
    logits = output[0]
    #labels = output[2]

    #val_loss += elementloss.item()
        
    logits = logits.detach().cpu().numpy()
    predictions.append(logits)
    actual.append(inputs['labels'])
    
#val_loss_arg = val_loss/len(val_dataloader)

#Axis=0: Along rows, Axis=1: Along columns

predictions = numpy.concatenate(predictions, axis = 0)
actual = numpy.concatenate(actual, axis = 0)

flattened_predictions = numpy.argmax(predictions, axis=1).flatten()
flattened_actual = actual.flatten()

#print(predictions, actual)

numcases = len(actual)

numcorrect = 0
for case in range(numcases):
    predcase = flattened_predictions[case]
    actualcase = flattened_actual[case]
    print(predcase, actualcase)
    if(predcase == actualcase):
        print("CORRECT")
        numcorrect += 1
    else:
        print("INCORRECT")

print("\n\nNUMCORRECT: " + str(numcorrect) + "/" + str(numcases) + " (" + str(round(100 * float(numcorrect)/float(numcases), 2)) + "%)")

comparison = pandas.DataFrame()

catlist = []
predlist = []

for number in flattened_actual:
    if(number == 0):
        catlist.append("General")
        predlist.append("Real Count")
    elif(number == 1):
        catlist.append("Wiki")
        predlist.append("Real Count")
    elif(number == 2):
        catlist.append("Devices & Integrations")
        predlist.append("Real Count")
    elif(number == 3):
        catlist.append("Apps & Clients")
        predlist.append("Real Count")
    elif(number == 4):
        catlist.append("SmartApps & Automations")
        predlist.append("Real Count")
    elif(number == 5):
        catlist.append("Developer Programs")
        predlist.append("Real Count")


for number in flattened_predictions:
    if(number == 0):
        catlist.append("General")
        predlist.append("Predicted")
    elif(number == 1):
        catlist.append("Wiki")
        predlist.append("Predicted")
    elif(number == 2):
        catlist.append("Devices & Integrations")
        predlist.append("Predicted")
    elif(number == 3):
        catlist.append("Apps & Clients")
        predlist.append("Predicted")
    elif(number == 4):
        catlist.append("SmartApps & Automations")
        predlist.append("Predicted")
    elif(number == 5):
        catlist.append("Developer Programs")
        predlist.append("Predicted")

comparison["Topics"] = catlist
comparison["Count"] = predlist

seaborn.countplot(x="Topics", hue="Count" ,data=comparison).set_title("RESULTS")

matplotlib.pyplot.show()



