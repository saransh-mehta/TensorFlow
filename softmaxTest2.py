#softmax demonstration
import numpy as np
def softmax():
    A=np.random.randn(100,5)
    #print(A)
    expA=np.exp(A)
    answer= expA/np.sum(expA,axis=1,keepdims=True)
    print(answer)
    print(answer.sum())
softmax()
    
