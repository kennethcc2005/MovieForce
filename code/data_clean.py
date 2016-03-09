import numpy as np
import pandas as pd
from scipy import sparse
import csv


df_movie = pd.read_csv('../data/movies.csv')
with open('Data/keywords_clean.list', 'r') as f:
    data = f.read().split('\n')
data_np = np.array(data)
movie_title = df_movie.title
movie_keyword = [s for s in data_np if any(xs in s for xs in movie_title)]
with open('../data/new_movie_keywords', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(movie_keyword)

df_key = pd.DataFrame(movie_keyword)
#add columns to name, year and key word
df_key['name'] = [x.split('\t')[0] for x in df_key[0]]
df_key['title'] = [x.split(')')[0] + ')' for x in df_key[0]]
df_key['keyword'] = [x.split('\t')[-1] for x in df_key[0]]
df_key['year'] = [df_key['name'][i][df_key['name'][i].find("(")+1:df_key['name'][i].find(")")] for i in xrange(df_key.shape[0])]
df_key.to_csv('../data/df_key_long',index = False)

m_col = []
a_col = []
for movie_t in movie_title:
    for k,v in movies.iteritems():
        if k in movie_t:
            for names in v:
                m_col.append(movie_t)
                a_col.append(names)
movie_actor_long = pd.concat([pd.Series(m_col), pd.Series(a_col)], axis=1)
movie_actor_long.columns = ['title','actor']
movie_actor_long.to_csv('../data/movie_actor_long_example',index = False)

df_key = pd.read_csv('../data/df_key_long')
df_actor = pd.read_csv('../data/movie_actor_long_example')
df_actor['count'] = 1


df_key_matrix = pd.read_csv('../data/df_key_matrix')
df_actor_matrix = pd.read_csv('../data/df_actor_matrix')

