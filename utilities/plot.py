import matplotlib.pyplot as plt

def loss(history):

    plt.plot(history.history['loss'],'b')
    plt.plot(history.history['val_loss'],'r')
    plt.title('Collaborative Filter Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Validation'],loc='upper right')
    plt.show()