import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set(color_codes=True)

categorycsv = pd.read_csv('categoryrank.csv')
topcategory = list(categorycsv.Category)[:5]
topnumber = list(categorycsv['No.'])[:5]

datacsv=pd.read_csv('data.csv')
replies=list(datacsv.Replies)
views=list(datacsv.Views)

Y = topnumber
X = [0, 1, 2, 3, 4]
X1 = [0, 1, 2, 3, 4]
width = 0.5
xticks = topcategory

fig=plt.figure()
fig.set_size_inches(15, 9)
ax = plt.bar(X, Y, width, color="#0000FF", alpha=0.5)
b = plt.xticks(X1, xticks, fontsize=12)

plt.xlim(-0.25, 4.75)
plt.ylabel('Number of posts', fontsize=12)
plt.title('Number of Top 5 Categories\n', fontsize=12)
plt.savefig('categorybar.png', dpi=300)

plt.figure()
sns.distplot(replies, kde=False, rug=True)
plt.xlabel('Replies', fontsize=12)
plt.ylabel('Number of posts', fontsize=12)
plt.title('Replies distribution\n', fontsize=12)
plt.savefig('reply1.png', dpi=300)
plt.figure()
sns.distplot(replies, kde=False, rug=True, hist_kws={"range": [0, 100]},bins=10)
plt.xlabel('Replies', fontsize=12)
plt.ylabel('Number of posts', fontsize=12)
plt.title('Replies distribution\n', fontsize=12)
plt.savefig('reply2.png', dpi=300)

plt.figure()
sns.distplot(views, kde=False, rug=True)
plt.xlabel('Views', fontsize=12)
plt.ylabel('Number of posts', fontsize=12)
plt.title('Views distribution\n', fontsize=12)
plt.savefig('view1.png', dpi=300)
plt.figure()
sns.distplot(views, kde=False, rug=False, hist_kws={"range": [0, 1000]},bins=10)
plt.xlim(0, 1000)
plt.xlabel('Views', fontsize=12)
plt.ylabel('Number of posts', fontsize=12)
plt.title('Views distribution\n', fontsize=12)
plt.savefig('view2.png', dpi=300)
