#-*-coding:utf-8-*-  

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import jieba
from PIL import Image
import numpy as np

text_from_file_with_apath = open(
    '~/Downloads/Demo/visualize/Title.txt').read()  # change the path here

wordlist_after_jieba = jieba.cut(text_from_file_with_apath, cut_all = False)
wl_space_split = " ".join(wordlist_after_jieba)

# abel_mask = np.array(Image.open("/Users/Cassandra/Downloads/test/circle.png"))#mask, parameter:  mask=abel_mask
# stopwords = set(STOPWORDS)
# stopwords.add("自己")

my_wordcloud = WordCloud(background_color="black", colormap="inferno",
                         width=1200, height=600, max_font_size=150).generate(wl_space_split)

plt.imshow(my_wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()
