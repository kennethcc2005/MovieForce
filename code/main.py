import numpy as np
import pandas as pd
from scipy import sparse
import csv
# import graphlab
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
import os
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image  
from sklearn.externals.six import StringIO
from sklearn import tree
from collections import Counter

df_key = pd.read_csv('../data/df_key_long',dtype='unicode')
df_actor = pd.read_csv('../data/movie_actor_long_example',dtype='unicode')
df_actor['count'] = 1
# df_key_matrix = pd.read_csv('../data/df_key_matrix')
# df_actor_matrix = pd.read_csv('../data/df_actor_matrix')

#Get the df keyword only
df_keyword = df_key[['title','keyword','count']]
# keyword_sf = graphlab.SFrame(df_keyword)
# m_als = graphlab.recommender.factorization_recommender.create(keyword_sf, user_id = 'keyword', item_id = 'title', target = 'count', solver='als')

#Use Knn to get cluters for 2 groups. And try using random forest to get highest feature importance
# knn = KNeighborsRegressor(n_neighbors=2)
# kmeans = KMeans(n_clusters=2)
# df_key_mat0 =df_key_matrix.fillna(0)
# kmeans.fit_predict(df_key_mat0.as_matrix())
# array_features = np.array(df_key_mat0.columns.values())
# for i, cluster in enumerate(kmeans.cluster_centers_):
#     idx = cluster.argsort()[0:10]
#     print i, array_features[idx]

#Find out the median year as a start point:
df_key['year2'] = [x[-5:-1] for x in df_key['name']]
df_key['year'] = [x if (len(str(x)) == 4) else i for i in df_key.year2 for x in df_key.year]
df_key['year'] = [x if (len(str(x)) == 4) else i[-5:-1] for i in df_key['name']]
df_key_matrix = df_key.reset_index().pivot_table(values='count', index='title', columns='keyword', aggfunc='mean')

#get the year matrix for the keyword
df_key_matrix['year'] = [x.split('(')[1][:-1] for x in df_key_matrix.index.values]
# pd.get_dummies(df_key_matrix['year'])

df_key_mat_ex = df_key_matrix[df_key_matrix.columns[df_key_matrix.sum()>5]]
clf = DecisionTreeClassifier(max_depth = 20)
clf.fit(df_key_mat_ex.values,np.array(df_key_mat_ex.index))

with open("../data/movie.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

#reduce the movies with limited keywords
df = df_key_mat_ex[df_key_mat_ex.sum(axis = 1) > 5]

yr_cnt = Counter(df_key_matrix['year']).most_common()
year_list = [int(i[0]) for i in yr_cnt if i[1]>5]



##################################
# Play the data more and more!!!!
df_key = pd.read_csv('../data/df_key_long',dtype={'title': 'S30','keyword': 'S20','count':'int'})
df_key_matrix = df_key.reset_index().pivot_table(values='count', index='title', columns='keyword', aggfunc='mean')
df_key_matrix['year'] = [x.split('(')[1][:-1] for x in df_key_matrix.index.values]
#Better to use this one!
yr_cnt = Counter(df_key_matrix['year']).most_common()
year_list = [i[0] for i in yr_cnt if i[1]>5]
df_key_matrix = df_key_matrix[df_key_matrix['year'].isin(year_list)]
df_key_matrix['year'] = df_key_matrix['year'].astype(int)
df_key_mat_ex = df_key_matrix[df_key_matrix.columns[df_key_matrix.sum()>5]].fillna(0)
df = df_key_mat_ex.copy()
year = df.pop('year')
df = df[df.sum(axis = 1) > 5]
df = df.join(year)
##################################
#Get 20 depth decision treeeeeeee!!!
clf = DecisionTreeClassifier(max_depth = 20)
clf.fit(df.values,np.array(df.index))
clf.score(df.values,np.array(df.index))

##################################
