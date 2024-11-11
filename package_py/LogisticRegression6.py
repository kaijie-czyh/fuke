import numpy as np  
class LogisticRegression:  
    def __init__(self,X,y, learning_rate=0.01, num_iterations=1000, solver='gd', tol=1e-4, C=1.0, verbose=False): 
        # X (numpy.ndarray): 特征矩阵，形状为 (样本数, 特征数)。
        # y (one_hot): 标签向量，形状为 (样本数,标签类型数)。
        self.labels=y.shape[1] #标签类型数
        # print("labels",self.labels)
        self.samples_num=X.shape[0] 
        self.features_num=X.shape[1]
        ones=np.ones((X.shape[0],1))
        self.dataX=np.c_[ones,X]
        self.y=y
        self.theta=np.zeros((self.dataX.shape[1], self.labels)) # 参数初始化
        self.learning_rate=learning_rate
        self.num_iterations=num_iterations
        self.solver=solver
        self.tol=tol
        self.C=C
        self.verbose=verbose
        self.loss_history=[]
        self.theta_history=[]
        self.fit()
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    def fit(self):
        if self.solver=='gd':
            self.gradient_descent()
        elif self.solver=='sgd':
            self.stochastic_gradient_descent()
        elif self.solver=='newton':
            self.newton_method()
    def gradient_descent(self):
        for i in range(self.num_iterations):
            # print("self.dataX.shape",self.dataX.shape) -> (样本数,特征数+1)
            # print("self.theta.shape",self.theta.shape) -> (特征数+1,标签类型数量)
            z=np.dot(self.dataX,self.theta)
            # print("z.shape",z.shape) -> (样本数,标签类型数量)

            y_pred=self.softmax(z)
            # print(y_pred.shape)
            # print("self.y",self.y.shape)
            loss=-np.sum(self.y*np.log(y_pred + 1e-9))/self.samples_num
            self.loss_history.append(loss)
            self.theta_history.append(self.theta)
            gradient=np.dot(self.dataX.T,(y_pred-self.y))/self.samples_num
            self.theta-=self.learning_rate*gradient
            if self.verbose and i%100==0:
                print('Iteration %d, Loss: %f'%(i,loss))
    def predict(self,X):
        ones=np.ones((X.shape[0],1))
        dataX=np.c_[ones,X]
        z=np.dot(dataX,self.theta)
        y_pred=self.softmax(z)
        # print("y_pred",y_pred)

        return np.argmax(y_pred,axis=1)
