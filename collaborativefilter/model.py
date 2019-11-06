from numpy import unique

from keras import backend as k
from keras.layers import Input,Embedding,Reshape,Dot
from keras.models import Model,load_model

from utilities import model

class CollaborativeFilter(object):
    def __init__(self):  
        
        self.model = None
        self.history = None
        self.init_settings = model.initialize()
        self.compile_settings = model.compile()
        self.fit_settings = model.fit()

    def initialize(self,user_shape,item_shape):

        user = Input(shape=self.init_settings['shape'])
        user_embedding = Embedding(input_dim=user_shape,output_dim=self.init_settings['embedding_size'],input_length=self.init_settings['input_length'])(user)
        reshaped_user = Reshape([self.init_settings['embedding_size']])(user_embedding)
    
        item = Input(shape=self.init_settings['shape'])
        item_embedding = Embedding(input_dim=item_shape,output_dim=self.init_settings['embedding_size'],input_length=self.init_settings['input_length'])(item)
        reshaped_item = Reshape([self.init_settings['embedding_size']])(item_embedding)
        
        dot = Dot(self.init_settings['axes'],normalize=self.init_settings['normalize'])([reshaped_user,reshaped_item])
        
        model = Model(inputs=[user,item],output=dot)
        
        print(model.summary())

        model.compile(**self.compile_settings)
        
        return model

    def fit(self,users,items,ratings):
        
        user_shape = unique(users).shape[0]
        item_shape = unique(items).shape[0]
    
        self.model = self.initialize(user_shape,item_shape)
        
        print('Fitting model ...') 
        self.history = self.model.fit(x=[users,items],y=ratings,**self.fit_settings)
    
    def predict(self,x):
        return self.model.predict(x)
     
    def save_model(self):
        print('saving model to disk....')
        self.model.save("./collaborativefilter/trained/model.h5")
        print('completed')

    def load_trained_model(self):
        print('loading model from disk...')
        return load_model("./collaborativefilter/trained/model.h5")
        print('completed')
