# NLP
Netflix Content Based Recommendation engine


# %% [code]

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# %% [code]


# %% [code]
import pandas as pd
df = pd.read_csv("../input/netflix_titles.csv")

# %% [code]
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import cufflinks as cf


# %% [code]
df.head()


# %% [code]
df.describe()

# %% [code]
df["date_added"]= pd.to_datetime(df["date_added"])
df['year_added']=df["date_added"].dt.year
df['month_added']=df['date_added'].dt.month
df.head()

# %% [code]
df['season_count']=df.apply(lambda x:x['duration'].split(" ")[0] if 'Season'  in x['duration'] else "",axis=1) #apply to each column
df['duration']=df.apply(lambda x:x['duration'].split(" ")[0] if 'Season' not in x['duration'] else "", axis=1)
df.head()



# %% [code]
from wordcloud import WordCloud , STOPWORDS , ImageColorGenerator
plt.rcParams['figure.figsize']=(13,13)
wordcloud=WordCloud(stopwords=STOPWORDS,background_color='black',width=1000,height=1000,max_words=121).generate(''.join(df['title']))
#Wordcloud function is used to make the cloud, remove stopwords and generate from df{title}
plt.imshow(wordcloud)
plt.axis('off')
plt.title('MOST POPULAR WORDS IN TITLE')
plt.show()

# %% [code]
import plotly.express as px
#x=count(df['type'] if df[type]==)
sr=pd.Series(df['type'])
x=sr.value_counts()
labels = 'Movies','TV Shows'
fig1, ax1 = plt.subplots()
ax1.pie(x,labels=labels,autopct='%1.1f%%')
plt.show()
x

# %% [code]
new_df=df[['title','director','cast','listed_in','description']]
new_df.head()
new_df.shape

# %% [code]
!pip install rake-nltk
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# %% [code]
new_df.dropna(inplace=True)
new_df.shape
blanks=[]
col=['title','director','cast','listed_in','description']
for i,col in new_df.iterrows():  # iterate over the DataFrame
    if type(col)==str:            # avoid NaN values
        if col.isspace():         # test 'review' for whitespace
            blanks.append(i)  
new_df.shape            

# %% [code]

new_df.drop(blanks, inplace=True)
new_df.shape   

# %% [code]
new_df['Keywords']=''
for index, row in new_df.iterrows():
    description = row['description']
    r= Rake()
    r.extract_keywords_from_text(description)
    key_words_dict_scores = r.get_word_degrees()
    row['Keywords']=list(key_words_dict_scores.keys())
new_df.drop(columns=['description'],inplace=True)
new_df.head()

# %% [code]
new_df['cast']=new_df['cast'].map(lambda x: x.split(',')[:3])
new_df['listed_in']=new_df['listed_in'].map(lambda x : x.lower().split(','))
new_df['director']=new_df['director'].map(lambda x: x.split(' '))
for index , row in new_df.iterrows():
    row['cast']=[x.lower().replace(' ','') for x in row['cast']]
    row['director']="".join(row['director']).lower()
    
    

# %% [code]
new_df.set_index('title', inplace = True)
new_df.head()

# %% [code]
new_df['bag_of_words'] = ''
columns = new_df.columns
columns

# %% [code]
new_df['bag_of_words']=''
columns=new_df.columns
for index, row in new_df.iterrows():
    words=''
    for col in columns:
        if col!='director':
            words=words+" ".join(row[col])+' '
        else:
            words=words+ row[col]+' '
            
    row['bag_of_words']=words


new_df.drop(columns=[col for col in  new_df.columns if col!= 'bag_of_words'],inplace=True)


# %% [code]
new_df.head()

# %% [code]
count=CountVectorizer()
count_matrix= count.fit_transform(new_df['bag_of_words'])
print(count_matrix)

# %% [code]
indices=pd.Series(new_df.index)
indices[:5]
idx = indices[indices == 'Automata'].index[0]
print(idx)


# %% [code]
cosine_sim=cosine_similarity(count_matrix,count_matrix)
cosine_sim

# %% [code]
def recommendations(Title, cosine_sim=cosine_sim):
    recommended_movies=[]
    idx=indices[indices==Title].index[0]
    score_series=pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top_10_indexes=list(score_series.iloc[1:11].index)
    for i in top_10_indexes:
        recommended_movies.append(list(new_df.index)[i])
    return recommended_movies

    

# %% [code]
recommendations('Rocky')
