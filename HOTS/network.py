import numpy as np
import matplotlib.pyplot as plt
from HOTS.layer import layer
from HOTS.timesurface import timesurface
from HOTS.stats import stats
from tqdm import tqdm
import os
import pickle

class network(object):
    """network is an Hierarchical network described in Lagorce et al. 2017 (HOTS).
    METHODS:
             .running -> runs the network from a loader and saves stream of events as output (learn=False), or a network with trained weights (learn=True)
             .get_fname -> returns the name of the network depending on its parameters
             .plotlayer -> plots the histogram of activation of the different layers ad associated kernels
             .plotconv -> plots the convergence of the layers during learning phase
             .plotactiv -> plots the activation map of each layer
             """

    def __init__(self,  name = 'hots',
                        timestr = None, # date of creation of the network 
                                        # (can add dataset name to discriminate models)
                        nbclust = (4,8,16), # architecture of the network (default=Lagorce2017)
                        # parameters of time-surfaces and datasets
                        tau = (1e1,1e2,1e3), #time constant for exponential decay in millisec
                        R = (2,4,8), # parameter defining the spatial size of the time surface
                        homeo = (.25,1), # parameters for homeostasis (None is no homeo rule)
                        camsize = (34,34), # size of the pixel grid that recorded the event stream
                        to_record = False
                ):
        self.name = name
        self.date = timestr
        if self.name == 'hots':
            # replicates methods from Lagorce et al. 2017
            algo, decay, krnlinit, homeo, sigma = 'lagorce', 'exponential', 'first', None, None
        elif self.name == 'homhots':
            # replicates methods from Grimaldi et al. 2021
            algo, decay, krnlinit, sigma = 'lagorce', 'exponential', 'rdn', None
            
        nbpolcam = 2 # number of polarities for the event stream as input of the network
        tau = np.array(tau)*1e3 # to enter tau in ms
        nblay = len(nbclust)
        if R is None:
            R = ((R,)*3)
        if to_record:
            self.stats = [[]]*nblay
        self.TS = [[]]*nblay
        self.L = [[]]*nblay
        self.stats = False
        if to_record:
            self.stats = [[]]*nblay
        for lay in range(nblay):
            if lay == 0:
                self.TS[lay] = timesurface(R[lay], tau[lay], camsize, nbpolcam, sigma, decay)
                self.L[lay] = layer(R[lay], nbclust[lay], nbpolcam, homeo, algo, krnlinit, camsize, to_record)
                if to_record:
                    self.stats[lay] = stats(nbclust[lay], camsize)
            else:
                self.TS[lay] = timesurface(R[lay], tau[lay], camsize, nbclust[lay-1], sigma, decay)
                self.L[lay] = layer(R[lay], nbclust[lay], nbclust[lay-1], homeo, algo, krnlinit, camsize, to_record)
                if to_record:
                    self.stats[lay] = stats(nbclust[lay], camsize)

##___________________________________________________________________________________________

    def running(self, loader, ordering, classes, train=True, learn=False, jitter=None, verbose=True):
        
        x_index = ordering.index('x')
        y_index = ordering.index('y')
        t_index = ordering.index('t')
        p_index = ordering.index('p')
        
        if learn:
            model, loaded = self.load_model(verbose)
            if loaded:
                self.L = model.L
                self.TS = model.TS
                if model.stats:
                    self.stats = model.stats
                return
        else:
            if train:
                output_path = f'../Records/output/train/{self.get_fname()}_{jitter}/'
            else: output_path = f'../Records/output/test/{self.get_fname()}_{jitter}/'

            if os.path.exists(output_path):
                if verbose: print(f'this dataset have already been processed, check at: \n {output_path}')
                return
            else:
                for classe in classes:
                    os.makedirs(output_path+f'{classe}')
            
        pbar = tqdm(total=len(loader))
        nb = 0
        for events, target in loader:
            no_output = 0
            events_output = np.zeros([4])
            events = events.squeeze()
            pbar.update(1)
            for i in range(len(self.L)):
                self.TS[i].spatpmat[:] = 0
                self.TS[i].iev, self.TS[i].x, self.TS[i].y, self.TS[i].t, self.TS[i].p = 0,0,0,0,0
                self.L[i].cumhisto[:] = 1
                if self.stats:
                    self.stats[i].actmap[:] = 0
            for iev in range(len(events)):
                x, y, t, p = int(events[iev][x_index].item()), int(events[iev][y_index].item()), int(events[iev][t_index].item()), int(events[iev][p_index].item())
                for lay in range(len(self.L)):
                    dic_prev = self.L[lay].kernel.copy()
                    timesurf = self.TS[lay].addevent(x, y, t, p)
                    if np.isnan(timesurf).sum()>0:
                        #self.plote()
                        print(iev)
                    if len(timesurf)>0:
                        p = self.L[lay].run(timesurf, learn)
                        if self.stats:
                            self.stats[lay].actmap[p,x,y] = 1
                            self.stats[lay].update(p, self.L[lay].kernel, timesurf, self.TS[lay].tau, dic_prev)
                        if lay==len(self.TS)-1:
                            events_output = np.vstack((events_output, np.array([x,y,t,p])))
                    else:
                        #no_output += 1
                        #print(f'{no_output} events did not reach the output layer, total number of events: {len(events)}', end='\r')
                        break
            if not learn and len(events_output.shape)>1:
                np.save(output_path+f'{classes[target]}/{nb}', events_output)
                nb+=1
        pbar.close()
        if learn:
            self.save_model()

    def get_fname(self):
        arch = [self.L[i].kernel.shape[1] for i in range(len(self.L))]
        R = [self.L[i].R for i in range(len(self.L))]
        tau = [np.round(self.TS[i].tau*1e-3,2) for i in range(len(self.TS))]
        f_name = f'{self.date}_{self.name}_{self.L[0].homeo}_{arch}_{tau}_{R}'
        return f_name

    def save_model(self):
        path = '../Records/models/'
        if not os.path.exists(path):
            os.makedirs(path)
        f_name = path+self.get_fname()+'.pkl'
        with open(f_name, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def load_model(self, verbose):
        loaded = False
        model = []
        path = '../Records/models/'
        f_name = path+self.get_fname()+'.pkl'
        if os.path.isfile(f_name):
            if verbose: print(f'loading a network with name:\n {f_name}')
            with open(f_name, 'rb') as file:
                model = pickle.load(file)
            loaded = True
        return model, loaded

    def sensformat(self,sensor_size):
        for i in range(1,len(self.TS)):
            self.TS[i].camsize = sensor_size
            self.TS[i].spatpmat = np.zeros((self.L[i-1].kernel.shape[1],sensor_size[0]+1,sensor_size[1]+1))
            self.stats[i].actmap = np.zeros((self.L[i-1].kernel.shape[1],sensor_size[0]+1,sensor_size[1]+1))
        self.TS[0].camsize = sensor_size
        self.TS[0].spatpmat = np.zeros((2,sensor_size[0]+1,sensor_size[1]+1))
        self.stats[0].actmap = np.zeros((2,sensor_size[0]+1,sensor_size[1]+1))


##___________________PLOTTING________________________________________________________________
##___________________________________________________________________________________________

    def plotlayer(self, maxpol=None, hisiz=2, yhis=0.3):
        '''
        '''
        N = []
        P = [2]
        R2 = []
        for i in range(len(self.L)):
            N.append(int(self.L[i].kernel.shape[1]))
            if i>0:
                P.append(int(self.L[i-1].kernel.shape[1]))
            R2.append(int(self.L[i].kernel.shape[0]/P[i]))
        if maxpol is None:
            maxpol=P[-1]

        fig = plt.figure(figsize=(16,9))
        gs = fig.add_gridspec(np.sum(P)+hisiz, np.sum(N)+len(self.L)-1, wspace=0.05, hspace=0.05)
        if self.L[-1].homeo:
            fig.suptitle('Activation histograms and associated time surfaces with homeostasis', size=20, y=0.95)
        else:
            fig.suptitle('Activation histograms and associated time surfaces for original hots', size=20, y=0.95)

        for i in range(len(self.L)):
            ax = fig.add_subplot(gs[:hisiz, int(np.sum(N[:i]))+1*i:int(np.sum(N[:i+1]))+i*1])
            plt.bar(np.arange(N[i]), self.L[i].cumhisto/np.sum(self.L[i].cumhisto), width=1, align='edge', ec="k")
            ax.set_xticks(())
            #if i>0:
                #ax.set_yticks(())
            ax.set_title('Layer '+str(i+1), fontsize=16)
            plt.xlim([0,N[i]])
            yhis = 1.1*max(self.L[i].cumhisto/np.sum(self.L[i].cumhisto))
            plt.ylim([0,yhis])

        #f3_ax1.set_title('gs[0, :]')
            for k in range(N[i]):
                vmaxi = max(self.L[i].kernel[:,k])
                for j in range(P[i]):
                    if j>maxpol-1:
                        pass
                    else:
                        axi = fig.add_subplot(gs[j+hisiz,k+1*i+int(np.sum(N[:i]))])
                        krnl = self.L[i].kernel[j*R2[i]:(j+1)*R2[i],k].reshape((int(np.sqrt(R2[i])), int(np.sqrt(R2[i]))))

                        axi.imshow(krnl, vmin=0, vmax=vmaxi, cmap=plt.cm.plasma, interpolation='nearest')
                        axi.set_xticks(())
                        axi.set_yticks(())
        plt.show()
        return fig

    def plotconv(self):
        fig = plt.figure(figsize=(15,5))
        for i in range(len(self.L)):
            ax1 = fig.add_subplot(1,len(self.stats),i+1)
            x = np.arange(len(self.stats[i].dist))
            ax1.plot(x, self.stats[i].dist)
            ax1.set(ylabel='error', xlabel='events (x'+str(self.stats[i].nbqt)+')', title='Mean error (eucl. dist) on '+str(self.stats[i].nbqt)+' events - Layer '+str(i+1))
        #ax1.title.set_color('w')
            ax1.tick_params(axis='both')

    def plotactiv(self, maxpol=None):
        N = []
        for i in range(len(self.L)):
            N.append(int(self.L[i].kernel.shape[1]))

        fig = plt.figure(figsize=(16,5))
        gs = fig.add_gridspec(len(self.L), np.max(N), wspace=0.05, hspace=0.05)
        fig.suptitle('Activation maps of the different layers', size=20, y=0.95)

        for i in range(len(self.L)):
            for k in range(N[i]):
                    axi = fig.add_subplot(gs[i,k])
                    axi.imshow(self.stats[i].actmap[k].T, cmap=plt.cm.plasma, interpolation='nearest')
                    axi.set_xticks(())
                    axi.set_yticks(())
                    
    def plotTS(self, maxpol=None):
        N = []
        for i in range(len(self.TS)):
            N.append(int(self.TS[i].spatpmat.shape[1]))

        fig = plt.figure(figsize=(16,5))
        gs = fig.add_gridspec(len(self.TS), np.max(N), wspace=0.05, hspace=0.05)
        fig.suptitle('Global TS of the different layers', size=20, y=0.95)

        for i in range(len(self.TS)):
            for k in range(N[i]):
                axi = fig.add_subplot(gs[i,k])
                axi.imshow(self.TS[i].spatpmat, cmap=plt.cm.plasma, interpolation='nearest')
                axi.set_xticks(())
                axi.set_yticks(())
    