import numpy as np
from TimeSurface import timesurface
import matplotlib.pyplot as plt

class layer(object):
    """layer makes the computations within a layer of the HOTS network based on the methods from Lagorce et al. 2017, Maro et al. 2020 or the Matching Pursuit algorithm.
    """
    
    def __init__(self, R, N_clust, nbpola, homeo, algo, krnlinit, to_record):
        self.to_record = to_record 
        self.R = R               
        self.homeo = homeo       # gives the parameters of the homeostasis rule (None if no homeostasis)
        self.algo = algo
        self.nbtrain = 0         # number of TS sent in the layer
        self.krnlinit = krnlinit # initialization of the kernels, can be 'rdn' (random) or 'first' (based on the first inputs)
        self.kernel = np.random.rand(nbpola*(2*R+1)**2, N_clust)
        self.kernel /= np.linalg.norm(self.kernel)
        self.cumhisto = np.ones([N_clust])
        
    def homeorule(self):
        ''' defines the homeostasis rule
        '''
        histo = self.cumhisto.copy()
        histo/=np.sum(histo)

        gain = np.exp(self.homeo[0]*self.kernel.shape[1]**self.homeo[1]*(1-histo*self.kernel.shape[1]))
        return gain
        
    
##____________DIFFERENT METHODS________________________________________________________
    
    def run(self, TS, learn):
        
        if self.krnlinit=='first':
            while self.nbtrain<self.kernel.shape[1]:
                self.kernel[:,self.nbtrain]=TS.T
                p = self.nbtrain
                self.nbtrain += 1
                return p

        simil = np.dot(TS,self.kernel)/(np.linalg.norm(TS)*np.linalg.norm(self.kernel))

        if self.homeo:
            gain = self.homeorule()
            closest_proto_idx = np.argmax(simil*gain)
        else:
            closest_proto_idx = np.argmax(simil)

        if learn:
            Ck = self.kernel[:,closest_proto_idx]
            alpha = 0.01/(1+self.cumhisto[closest_proto_idx]/20000)
            Ck_t = Ck + alpha*simil[closest_proto_idx]*(TS - Ck)
            #Ck_t = Ck + alpha*(TS - simil[closest_proto_idx]*Ck)
            self.kernel[:,closest_proto_idx] = Ck_t

        p = closest_proto_idx
        self.cumhisto[closest_proto_idx] += 1
        if learn:
            self.nbtrain += 1
        
        return p
    
##____________PLOTTING_________________________________________________________________________
    
    def plotdicpola(lay, pola, R):
        fig = plt.figure(figsize=(15,5))
        fig.suptitle("Dictionary after {0} events" .format(lay.nbtrain))
        for n in range(len(lay.kernel[0,:])):
            for p in range(pola):
                sub = fig.add_subplot(pola,len(lay.kernel[0,:]),n+len(lay.kernel[0,:])*p+1)
                dico = np.reshape(lay.kernel[p*(2*R+1)**2:(p+1)*(2*R+1)**2,n], [int(np.sqrt(len(lay.kernel)/pola)), int(np.sqrt(len(lay.kernel)/pola))])
                sub.imshow((dico))
                sub.axes.get_xaxis().set_visible(False)
                sub.axes.get_yaxis().set_visible(False)
        plt.show()