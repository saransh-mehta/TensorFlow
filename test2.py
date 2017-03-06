#tensorflow test2
#importing packages
import tensorflow as tf
import numpy as np

#creating training data

xData=np.random.rand(100).astype(np.float32) #training set
yData=xData*0.2+0.3                 #target values

#creating structure
weights=tf.Variable(np.random.rand(1))    #gives no uniformly between 0,1 in given form
biases=tf.Variable(np.zeros(1))

y=weights*xData+biases

loss=tf.reduce_mean(np.square(y-yData))
optimizer=tf.train.GradientDescentOptimizer(0.4)
train=optimizer.minimize(loss)

#initializing variables
init=tf.global_variables_initializer()
#session

sess=tf.Session()
sess.run(init)

for i in range(204):
    sess.run(train)
    if i%20==0:
        print(i,sess.run(weights),sess.run(biases))


