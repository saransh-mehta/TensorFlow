# test a nn model learning through tensor flow

import tensorflow as tf
import numpy as np
#generating data (training set)

x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3  #targeted output for training

#generating nn structure through tf

weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))  #set weights and bias as the variables
biases=tf.Variable(tf.zeros([1]))

y=weights*x_data+biases   #y on which improvement will be done

loss=tf.reduce_mean(tf.square(y-y_data))

train=tf.train.GradientDescentOptimizer(0.4).minimize(loss)

#some tensorflow shit

init=tf.initialize_all_variables()  #in tf we need to do this shit to initialize the variables we created
sess=tf.Session()
sess.run(init)

for i in range(201):
    sess.run(train)   #powers up all other things
    if i%20==0:
        print(i,sess.run(weights),sess.run(biases))



