import numpy as np


class KNN:
    def __init__(self, train_data, train_goal, theta):
        self.train_data = train_data
        self.train_goal = train_goal
        self.theta = np.ones(train_data.shape[1])
        self.num=train_data.shape[1]

    def dis(self,x,data):
        l=self.num
        distance=0
        for i in range(l):
            distance+=self.theta[i]*(x[i]-data[i])*(x[i]-data[i])
        return distance
    
    def predict_data(self,data,k):
        t_data=np.vstack((self.train_data.T,self.train_goal)).T
        knn_data=sorted(t_data,key=lambda x:self.dis(x,data))
        rate=0
        for i in range(k):
            rate+=knn_data[i][-1]
        if 2*rate>k:
            return 1
        else:
            return 0





