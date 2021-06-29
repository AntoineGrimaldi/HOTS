import numpy as np
from TimeSurface import timesurface
import matplotlib.pyplot as plt

class layer(object):
    """layer makes the computations within a layer of the HOTS network based on the methods from Lagorce et al. 2017, Maro et al. 2020 or the Matching Pursuit algorithm.
    """
    
    def __init__(self, R, N_clust, nbpola, homeo, algo, krnlinit, output, to_record):
        self.to_record = to_record 
        self.R = R               
        self.algo = algo         # can be 'lagorce','maro', 'mpursuit' regarding the method
        self.homeo = homeo       # boolean indicating if homeostasis is used or not
        self.nbtrain = 0         # number of TS sent in the layer
        self.krnlinit = krnlinit # initialization of the kernels, can be 'rdn' (random) or 'first' (based on the first inputs)
        self.output = output     # defines output values - among ['se', 'sv', 'me', 'mv'] for 'single event', 'single value', 'multiple event' or 'multiple values'
        self.kernel = np.random.rand(nbpola*(2*R+1)**2, N_clust)
        self.kernel /= np.linalg.norm(self.kernel)
        self.cumhisto = np.ones([N_clust])
        
        if algo == 'maro':
            self.last_time_activated = np.zeros(N_clust)
        
    def homeorule(self):
        ''' defines the homeostasis rule
        '''
        histo = self.cumhisto.copy()
        histo/=np.sum(histo)

        if self.algo=='mpursuit':
            mu = 1
            gain = np.log(histo)/np.log(mu/self.kernel.shape[1])
        #__________________________________________
        else:
            homparam = [.25, 1]
            gain = np.exp(homparam[0]*(self.kernel.shape[1]**homparam[1])*(1-histo*self.kernel.shape[1]))
        return gain
    
    def run(self, TS, learn):
        
        if self.algo=='lagorce':
            h, temphisto = self.lagorce(TS, learn)
        elif self.algo=='mpursuit':
            h, temphisto = self.mpursuit(TS, learn)
        elif self.algo=='maro':
            h, temphisto = self.maro(TS, learn)
        self.cumhisto += temphisto
            
        if learn:
            self.nbtrain += 1

        return h
    
    
##____________DIFFERENT METHODS________________________________________________________
    
    def lagorce(self, TS, learn):
        
        h = np.zeros([self.kernel.shape[1]])
        
        if self.krnlinit=='first':
            while self.nbtrain<self.kernel.shape[1]:
                self.kernel[:,self.nbtrain]=TS.T
                h[self.nbtrain] = 1
                temphisto = h.copy()
                return h, temphisto

        simil = np.dot(TS,self.kernel)/(np.linalg.norm(TS)*np.linalg.norm(self.kernel))
        
        gain = np.ones([len(self.cumhisto)])
        if self.homeo:
            gain *= self.homeorule()
            closest_proto_idx = np.argmax(simil*gain)
        else:
            closest_proto_idx = np.argmax(simil)

        if learn:
            pk = self.cumhisto[closest_proto_idx]
            Ck = self.kernel[:,closest_proto_idx]
            alpha = 0.01/(1+pk/20000)
            Ck_t = Ck + alpha*(TS - simil[closest_proto_idx]*Ck)
            self.kernel[:,closest_proto_idx] = Ck_t

        if self.output=='se' or 'me':
            h[closest_proto_idx] = 1
            temphisto = h.copy()
        else:
            h[closest_proto_idx] = simil[closest_proto_idx]
            temphisto = np.ceil(h.copy())
        
        return h, temphisto
    
    def maro(self, TS, learn):
        
        h = np.zeros([self.kernel.shape[1]])
        
        if self.krnlinit=='first':
            while self.nbtrain<self.kernel.shape[1]:
                self.kernel[:,self.nbtrain]=TS.T
                h[self.nbtrain] = 1
                temphisto = h.copy()
                return h, temphisto

        simil = np.dot(TS,self.kernel)/(np.linalg.norm(TS)*np.linalg.norm(self.kernel))
        
        if self.homeo:
            gain = self.homeorule()
            closest_proto_idx = np.argmax(simil*gain)
        else:
            closest_proto_idx = np.argmax(simil)
            
        if learn:
            pk = self.cumhisto[closest_proto_idx]
            Ck = self.kernel[:,closest_proto_idx]
            self.last_time_activated[closest_proto_idx] = self.nbtrain
            alpha = 1/(1+pk)
            Ck_t = Ck + alpha*(TS - simil[closest_proto_idx]*Ck)
            self.kernel[:,closest_proto_idx] = Ck_t
            
            critere = (self.nbtrain-self.last_time_activated) > 10000
            critere2 = self.cumhisto < 25000
            if np.any(critere2*critere):
                cri = self.cumhisto[critere] < 25000
                idx_critere = np.arange(0, self.kernel.shape[1])[critere][cri]
                for idx_c in idx_critere:
                    Ck_t = Ck + 0.2*simil[closest_proto_idx]*(TS.T-Ck)
                    self.kernel[:,idx_c] = Ck_t
        
        if self.output=='se' or 'me':
            h[closest_proto_idx] = 1
            temphisto = h.copy()
        else:
            h[closest_proto_idx] = simil[closest_proto_idx]
            temphisto = np.ceil(h.copy())
        
        return h, temphisto
    
    def mpursuit(self, TS, learn):
        alpha = 1
        eta = 0.005
        h = np.zeros([self.kernel.shape[1]]) # sparse vector
        temphisto = np.zeros([len(h)])
        corr = np.dot(TS,self.kernel)
        Xcorr = np.dot(self.kernel.T, self.kernel)
        if self.homeo:
            gain = self.homeorule()
        while np.max(corr)>0: # here, Xcorr has relatively high values, meaning clusters are correlated. With the update rule of the MP, coefficients of corr can get negative after few iterations. This criterion is used to stop the loop
            if self.homeo:
                ind = np.argmax(corr*gain)
            else:
                ind = np.argmax(corr)
            h[ind] = corr[ind].copy()/Xcorr[ind,ind]
            corr -= alpha*h[ind]*Xcorr[:,ind]
            if learn:
                self.kernel[:,ind] = self.kernel[:,ind] + eta*h[ind]*(TS.T-self.kernel[:,ind])
                self.kernel[:,ind] = self.kernel[:,ind]/np.sqrt(np.sum(self.kernel[:,ind]**2))
            temphisto[ind] += 1
            
        if self.output=='se':
            h = h*(h==np.max(h))>0
        elif self.output=='me':
            h = h>0
        elif self.output=='sv':
            h = h*(h==np.max(h))
            
        return h, temphisto
    
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