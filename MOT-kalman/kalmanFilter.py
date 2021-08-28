#################### Import Section of the code #############################

try:
    import numpy as np

    #import GPU Acceleration functions
    from gpu_FMT import *       #FMT(a, b, M, N, K)
    from gpu_matmul import *    #matmul(a, b, M, N, K) 
    from gpu_add import *       #matrixAdd(a, b, M, N)
    from gpu_subtract import *  #matrixSub(a, b, M, N)
    from gpu_transpose import * #transpose_kernel(inpt, M, N)
except Exception as e:
    print(e, "\nPlease Install the package")

#################### Import Section ends here ################################
BLK_SIZE = 32

class KalmanFilter(object):
    M = 5
    N = 32
    K = 33
    """docstring for KalmanFilter"""

    def __init__(self, dt=1, stateVariance=1, measurementVariance=1,
                 method="Velocity"):
        super(KalmanFilter, self).__init__()
        self.method = method
        self.stateVariance = stateVariance
        self.measurementVariance = measurementVariance
        self.dt = dt
        self.initModel()

    """init function to initialise the model"""

    def initModel(self):
        if self.method == "Accerelation":
            self.U = 1
        else:
            self.U = 0
    
        self.A = np.matrix([[1, self.dt, 0, 0], [0, 1, 0, 0],
                            [0, 0, 1, self.dt],  [0, 0, 0, 1]])
        self.A.view()

        self.B = np.matrix([[self.dt**2/2], [self.dt], [self.dt**2/2],
                            [self.dt]])

        self.H = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0]])
        
        '''
        self.A = np.random.rand(1500, 1500)
        self.A.view()
        self.B = np.random.rand(1500, 1)
        self.H = np.random.rand(3000, 1500)
        '''
        self.P = np.matrix(self.stateVariance*np.identity(self.A.shape[0]))
        self.R = np.matrix(self.measurementVariance*np.identity(
            self.H.shape[0]))
        
        self.Q = np.matrix([[self.dt**4/4, self.dt**3/2, 0, 0],
                            [self.dt**3/2, self.dt**2, 0, 0],
                            [0, 0, self.dt**4/4, self.dt**3/2],
                            [0, 0, self.dt**3/2, self.dt**2]])
        
        #self.Q = np.random.rand(1500, 1500)
        self.erroCov = self.P
        self.state = np.matrix([[0], [1], [0], [1]])
        #self.state = np.random.rand(1500, 1)
    """Predict function which predicst next state based on previous state"""

    def predict(self):
        AX = matmul(self.A, self.state, self.A.shape[0], self.state.shape[1], self.A.shape[1]) 
        BU = self.B*self.U
        #BU = matmul(self.B, self.U, self.B.shape[0], self.U.shape[1], self.B.shape[1])
        self.predictedState = matrixAdd(AX, BU, AX.shape[0], AX.shape[1])
        #self.predictedState = self.A*self.state + self.B*self.U
        self.predictedState.view()
        #print(x)
        
        #P = error Covariance matrix
        AP = matmul(self.A, self.erroCov, self.A.shape[0], self.erroCov.shape[1], self.A.shape[1])
        APA_T = FMT(AP, self.A, AP.shape[0], self.A.shape[1], AP.shape[1])
        self.predictedErrorCov = matrixAdd(APA_T, self.Q, APA_T.shape[0], APA_T.shape[1])  
        #self.predictedErrorCov = self.A*self.erroCov*self.A.T + self.Q
        temp = np.asarray(self.predictedState)
        return temp[0], temp[2]

    """Correct function which correct the states based on measurements"""

    def correct(self, currentMeasurement):
        
        #K =Pk Hk T [Hk Pk Hk T + Ek] -1
        predP = self.predictedErrorCov
        predPH_T = FMT(predP, self.H, predP.shape[0], self.H.shape[1], predP.shape[1])
        HpredP = matmul(self.H, predP, self.H.shape[0], predP.shape[1], self.H.shape[1])
        HpredPH_T = FMT(HpredP, self.H, HpredP.shape[0], self.H.shape[1], HpredP.shape[1])
        inter_1 = matrixAdd(HpredPH_T, self.R, HpredPH_T.shape[0], HpredPH_T.shape[1])
        inverse = np.linalg.pinv(inter_1)
        self.kalmanGain = matmul(predPH_T, inverse, predPH_T.shape[0], inverse.shape[1], predPH_T.shape[1])
        #self.kalmanGain = self.predictedErrorCov*self.H.T*np.linalg.pinv(self.H*self.predictedErrorCov*self.H.T+self.R)
        
        #xk =xk + K [zk – Hkxk ]
        HpredX = matmul(self.H, self.predictedState, self.H.shape[0], self.predictedState.shape[1], self.H.shape[1]) 
        inter_2 = matrixSub(currentMeasurement, HpredX, currentMeasurement.shape[0], currentMeasurement.shape[1])
        Kinter_2 = matmul(self.kalmanGain, inter_2, self.kalmanGain.shape[0], inter_2.shape[1], self.kalmanGain.shape[1])
        self.state = matrixAdd(self.predictedState, Kinter_2, self.predictedState.shape[0], self.predictedState.shape[1])
        #self.state = self.predictedState + self.kalmanGain*(currentMeasurement  - (self.H*self.predictedState))
        
        #Pk = [I – K Hk]
        I = np.identity(self.P.shape[0])
        KH = matmul(self.kalmanGain, self.H, self.kalmanGain.shape[0], self.H.shape[1], self.kalmanGain.shape[1])
        inter_3 = matrixSub(I, KH, I.shape[0], I.shape[1])
        self.erroCov = matmul(inter_3, self.predictedErrorCov, inter_3.shape[0], self.predictedErrorCov.shape[1], inter_3.shape[1])
        #self.erroCov = (np.identity(self.P.shape[0]) - self.kalmanGain*self.H)*self.predictedErrorCov

