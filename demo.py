from utilities import data,model,plot

from sklearn.metrics import mean_absolute_error
import numpy as np 

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE

from collaborativefilter.model import CollaborativeFilter

import matplotlib.pyplot as plt 

def embeddings(cf,layer,labels):

    embedding = cf.model.get_weights()[layer]
    embedding = embedding[2:]
    
    labels = list(labels)

    distance_matrix = euclidean_distances(embedding)

    tnse = TSNE(n_components=2,random_state=0,n_iter=10000,perplexity=15)
    
    np.set_printoptions(suppress=True)
    
    T = tnse.fit_transform(distance_matrix)
    
    plt.figure(figsize=(20,10))
    plt.scatter(T[:,0],T[:,1])
    
    for label, x, y in zip(labels, T[:, 0], T[:, 1]):
        
        plt.annotate(label, xy=(x, y))
    
    plt.show()

def main():
    
    PATH = './utilities/ratings.csv'     
    
    data_ = data.load(PATH)
    data_ = data.transform(data_)
    train,test = data.split(data_)
    
    #test samples
    user_ids = list(test.userId)
    movie_ids = list(test.movieId)
    ratings = list(test.rating) 
    
    cf = CollaborativeFilter()
    trained_model = cf.load_trained_model()
    
    #first couple predictions for given user
    user_id = np.array([user_ids[0]])
    for i in range(10):
        movie_id = np.array([movie_ids[i]])
        prediction = trained_model.predict([user_id,movie_id])  
        print(ratings[i],prediction)
        
    #cf.fit(train.userId,train.movieId,train.rating)
    #cf.save_model()
    
    #user_embedding
    #embeddings(collaborativefilter,0,list(train.userId))

    #plot.loss(collaborativefilter.history)
         
    #predicting ratings for test sample 
    #predicted = collaborativefilter.predict([test_user_id,test_movie_id])
    #mse = mean_absolute_error(test_rating,predicted)
    #print(mse)    

if __name__=='__main__':
    main()

    
