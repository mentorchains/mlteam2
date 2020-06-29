import pandas as pd
from collections import defaultdict

my_cate = pd.read_csv('categoryrank.csv')
tag=list(my_cate.Category)
label = list(my_cate.Rank)
my_csv = pd.read_csv('data.csv')
data = my_csv[["tag", "starter_content"]]
print(data.head())

# Select column (can be A,B,C,D)
col = 'tag'

# Find and replace values in the selected column
data[col] = data[col].replace(tag, label)
print(data.head())

#keep only top 5 tags
df=data[data['tag']<=5]

print(df.head())

df.to_csv(r'./data4.csv', index=False, header=True)

# # data.head(500).to_csv(r'./data2.csv',index=False, header=True)
# cate = defaultdict(int)
# data2=data.head(500)
# tags=list(data2.tag)
# for i in range(500):
#     cate[str(tags[i])] += 1

# for k, v in sorted(cate.items(), key=lambda item: item[1], reverse=True):
#     print(k, ':', v)
# # print(len(data2))
