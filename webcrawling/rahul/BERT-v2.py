from bs4 import BeautifulSoup
from collections import defaultdict
import csv
import math
import matplotlib
import numpy
import inspect
from keras.preprocessing.sequence import pad_sequences #Requires TensorFlow installation
import pandas
import seaborn
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel, AdamW, get_linear_schedule_with_warmup
import random
import re
import requests
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#V2: Categories are the same as the article tags.

#Walkthrough Link: https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03

categoriesdict = defaultdict(int)
categories = []
bertTokens = []
bertIds = []
tags = []
numtags = []
sentencetags = []
num = 0

#Initiate the Bert Tokenizer
bertTokenize = BertTokenizer.from_pretrained('bert-base-uncased')

seed = 24

url = []

base = "https://community.smartthings.com"
page = "/latest?no_definitions=true&page="

#Go through infinite scrolling
for num in range(100):
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

#Assign each topic to a number
catnumber = 0
firstcat = ""
for element in tags:
    if(catnumber == 0):
        firstcat = element
        categoriesdict[element] = 0
        categories.append(element)
        catnumber += 1
    elif(categoriesdict[element] == 0 and element != firstcat):
        categoriesdict[element] = catnumber
        categories.append(element)
        catnumber += 1
    
    numtags.append(categoriesdict[element])

numcategories = len(categoriesdict)

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
    sentences = list(tokenizer.split_into_sentences(text))
   
    #Further formatting...
    for sentence in sentences:
        #Make whole sentence lowercase
        lowers = sentence.lower()
        #Remove punctuation
        s = re.sub(r'[^\w\s]', '', lowers)
        #Add necessary start/stop tokens for BERT
        sformatted = "[CLS] " + s + " [SEP]"

        #Use the Bert Tokenizer
        bertTokens.append(bertTokenize.tokenize(sformatted))

        #Add a label based on article category
        sentencetags.append(numtags[completed])

    #Go through n articles
    #if(completed == 10):
    #    break

    completed += 1
    print(str(completed) + "/" + str(len(url)) + " completed.")

#Create ID list for the tokens
for element in bertTokens:
    bertIds.append(bertTokenize.convert_tokens_to_ids(element))

#Pad the ID list so that all elements have a length of 128
max_len = 128
bertIds = pad_sequences(bertIds, maxlen=max_len, padding="post")

#Split data for training and testing (validation)
train_input_array, val_input_array, train_label, val_label = train_test_split(bertIds, sentencetags, test_size=0.15, random_state=seed)

#TrainTestSplit returns numpy arrays, so they must be converted to lists for BatchEncodePlus
train_input = train_input_array.tolist()
val_input = val_input_array.tolist()

#Special Tokens added (CLS/SEP), Want attention masks, Pad to 128 words, Already tokenized, Return pytorch tensors
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

print(categoriesdict)

comparison = pandas.DataFrame()

catlist = []
predlist = []

for number in flattened_actual:
    #catlist.append(categories[number])
    catlist.append("C"+str(number))
    predlist.append("Real Count")

for number in flattened_predictions:
    #catlist.append(categories[number])
    catlist.append("C"+str(number))
    predlist.append("Predicted")

comparison["Topics"] = catlist
comparison["Count"] = predlist

seaborn.countplot(x="Topics", hue="Count" ,data=comparison).set_title("RESULTS")

matplotlib.pyplot.show()


