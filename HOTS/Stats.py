import numpy as np

class stats(object):
    """ """

    def __init__(self, N, camsize):
        self.nbqt = 1000
        self.count = 0
        self.dist_cum = 0
        self.dist = []
        self.actmap = np.zeros([N,camsize[0]+1,camsize[1]+1])
        self.delta_wt = np.zeros([4])

    def update(self, p, dic, X, tau, dic_prev):
            dist = np.linalg.norm(X - dic[:,p])
            self.dist_cum += dist
            dt = -tau*np.log(X)
            dt_krnl = -tau*np.log(dic[:,p])
            dw = dic[:,p]-dic_prev[:,p]
            
            self.delta_wt = np.vstack((self.delta_wt, np.array([dw,dt,dt_krnl, dic_prev[:,p]]).T))

            self.count += 1
            if self.count==self.nbqt:
                self.dist.append(self.dist_cum/self.nbqt)
                self.dist_cum = 0
                self.count = 0
