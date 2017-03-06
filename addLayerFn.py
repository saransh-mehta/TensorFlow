# function to create a new layer

import tensorflow as tf

def addLayer(inputs,prvLyrNo,curLyrNo,actFun=None):
    
    #making weights and biases
    
    weights=tf.Variable(tf.random_normal([prvLyrNo,curLyrNo]))
    biases=tf.Variable(tf.zeroes(1,curLyrNo))

    #calculating Wp+b

    y=tf.matmul(weights,inputs)+biases

    #activation function, if a hidden layer then default is None

    if actFun==None:
        outputs=y
    else:
        outputs=actFun(y)
    return y
        
    
    
