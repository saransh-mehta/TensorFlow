# tensorflow mnist data classification
import tensorflow as tf

#importing data from tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#now we have 55000 2d matrices of 28x28 =784 pixels
# we ll align them as array of 784 n classify into 10 classes

def addLayer(inputs,prvLyrNo,curLyrNo,actFun=None):
    
    #making weights and biases

    weights=tf.Variable(tf.random_normal([prvLyrNo,curLyrNo]),name="W")

    biases=tf.Variable(tf.zeros([1,curLyrNo])+0.1,name="biases")

    #calculating Wp+b


    y=tf.matmul(inputs,weights)+biases

    #activation function, if a hidden layer then default is None

    if actFun is None:
        outputs=y
    else:
        outputs=actFun(y)
    return outputs

#function for calculating correct/total
def compAccuracy(t_xs,t_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:t_xs})
    correctPrediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(t_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:t_xs,ys:t_ys})
    return result

# data is already prepared

#placeholder

xs = tf.placeholder(tf.float32,[None,784]) #allowing all data through None n vertical 784 pixels

ys = tf.placeholder(tf.float32,[None,10]) # 10 classes ro be classified in

prediction = addLayer(xs,784,10,tf.nn.softmax)

#prediction will come like [0.001, 0.1, 0.03, 0.65,...] which means [0,0,0,1..]

# cost function for softmax
crossEntropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),axis=1))

#training step
train = tf.train.GradientDescentOptimizer(0.5).minimize(crossEntropy)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(1000):

    # using sochastic gradient descent

    batch_xs,batch_ys = mnist.train.next_batch(1000)
    sess.run(train,feed_dict = {xs:batch_xs,ys:batch_ys})
    
    if i%50 ==0:

        print(compAccuracy(mnist.test.images,mnist.test.labels))
    

