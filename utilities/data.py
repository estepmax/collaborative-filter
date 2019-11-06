import pandas as pd 

from sklearn.model_selection import train_test_split 

PATH = 'c:/enter/your/directory'

def load(PATH=PATH):
    return pd.read_csv(PATH)
    
def transform(data):
 
    movies = data.movieId.unique()
    id_to_index = {m:i for i,m in enumerate(movies)}
    
    users = data.userId.unique()
    id_to_index_ = {u:i for i,u in enumerate(users)}

    data['movieId'] = data['movieId'].apply(lambda x: id_to_index[x])
    data['userId'] = data['userId'].apply(lambda x: id_to_index_[x])

    return data

def split(df,cutoff=0.2,random_state=20):
    return train_test_split(df,test_size=cutoff,random_state=random_state) 

