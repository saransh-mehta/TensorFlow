#testing tensorflow placeholders
import tensorflow as tf
import numpy as np

input1=tf.placeholder(np.float32)

input2=tf.placeholder(np.float32)

output=tf.mul(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:np.random.rand(1).astype(np.float32),input2:np.random.rand(1).astype(np.float32)}))
