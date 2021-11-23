import numpy as np
import matplotlib.pyplot as plt
from Layer import layer
from TimeSurface import timesurface
from Stats import stats
from tqdm import tqdm
import tonic
import os
import pickle
from torch import Generator
from torch.utils.data import SubsetRandomSampler, DataLoader

class network(object):
    """network is an Hierarchical network described in Lagorce et al. 2017 (HOTS). It loads event stream with the tonic package.
    METHODS: .load -> loads datasets thanks to tonic package built on Pytorch.
             .learning1by1 -> makes the unsupervised clustering of the different layers 1 layer after the other
             .learningall -> makes the online unsupervised clustering of the different layers
             .running -> run the network and output either an averaged histogram for each class (train=True), either an histogram for each digit/video as input (train=False) either the stream of events as output of the last layer (LR=True)
             .run -> computes the run of an event as input of the network
             .get_fname -> returns the name of the network depending on its parameters
             .plotlayer -> plots the histogram of activation of the different layers ad associated kernels
             .plotconv -> plots the convergence of the layers during learning phase
             .plotactiv -> plots the activation map of each layer
             """

    def __init__(self,  name = 'hots',
                        timestr = None, # date of creation of the network
                        nbclust = [4, 8, 16], # architecture of the network (default=Lagorce2017)
                        # parameters of time-surfaces and datasets
                        tau = [1e1, 1e2, 1e3], #time constant for exponential decay in millisec
                        R = [2, 4, 8], # parameter defining the spatial size of the time surface
                        to_record = True
                ):
        self.name = name
        self.date = timestr
        if self.name == 'hots':
            # replicates methods from Lagorce et al. 2017
            algo, decay, krnlinit, homeo, sigma, output = 'lagorce', 'exponential', 'rdn', False, None, 'se'
        elif self.name == 'homhots':
            # replicates methods from Grimaldi et al. 2021
            algo, decay, krnlinit, homeo, sigma, output = 'lagorce', 'exponential', 'rdn', True, None, 'se'
            
        nbpolcam = 2 # number of polarities for the event stream as input of the network
        camsize = [34,34] # size of the pixel grid that recorded the event stream
        tau = np.array(tau)*1e3 # to enter tau in ms
        nblay = len(nbclust)
        if to_record:
            self.stats = [[]]*nblay
        self.TS = [[]]*nblay
        self.L = [[]]*nblay
        for lay in range(nblay):
            if lay == 0:
                self.TS[lay] = timesurface(R[lay], tau[lay], camsize, nbpolcam, sigma, decay)
                self.L[lay] = layer(R[lay], nbclust[lay], nbpolcam, homeo, algo, krnlinit, output, to_record)
                if to_record:
                    self.stats[lay] = stats(nbclust[lay], camsize)
            else:
                self.TS[lay] = timesurface(R[lay], tau[lay], camsize, nbclust[lay-1], sigma, decay)
                self.L[lay] = layer(R[lay], nbclust[lay], nbclust[lay-1], homeo, algo, krnlinit, output, to_record)
                if to_record:
                    self.stats[lay] = stats(nbclust[lay], camsize)

##___________________________________________________________________________________________

    def running(self, loader, indices, classes, learn=False, verbose = True):
        
        x_index = indices.index('x')
        y_index = indices.index('y')
        t_index = indices.index('t')
        p_index = indices.index('p')
        
        pbar = tqdm(total=len(loader))
        
        for events, target in loader:
            pbar.update(1)
            for i in range(len(self.L)):
                self.TS[i].spatpmat[:] = 0
                self.TS[i].iev = 0
                self.L[i].cumhisto[:] = 1
                #self.stats[i].actmap[:] = 0
            for iev in range(N_max):
                p = np.zeros([self.TS[0].spatpmat.shape[0]])
                x, y, t, p[int(events[0][iev][p_index].item())] = int(events[0][iev][x_index].item()), int(events[0][iev][y_index].item()), int(events[0][iev][t_index].item()), 1
                for lay in range(len(self.L)):
                    timesurf = self.TS[lay].addevent(x, y, t, p)
                    if len(timesurf)>0:
                        p = self.L[lay].run(timesurf, learn)
                    else:
                        break
        pbar.close()
            

    def learningall(self, nb_digit=10, train=True, dataset='nmnist', diginit=True, ds_ev=None, maxevts=None, kfold = None, kfold_ind = None, outstyle='histo', verbose=True):

        model = self.load_model(dataset, verbose)
        if model:
            return model
        else:
            loader, ordering, classes = self.load(dataset)
            nbclass = len(classes)
            pbar = tqdm(total=nb_digit*nbclass)
            nbloadz = np.zeros([nbclass])
            while np.sum(nbloadz)<nb_digit*nbclass:
                if diginit:
                    for i in range(len(self.L)):
                        self.TS[i].spatpmat[:], self.TS[i].iev = 0, 0
                events, target = next(iter(loader))
                if nbloadz[target]<nb_digit:
                    nbloadz[target]+=1
                    pbar.update(1)
                    if ds_ev:
                        events = events[:,::ds_ev,:]
                    if maxevts:
                        N_max = min(maxevts, events.shape[1])
                    else:
                        N_max = events.shape[1]
                    if dataset=='cars':
                        size_x = max(events[0,:,ordering.find("x")])-min(events[0,:,ordering.find("x")])+1
                        size_y = max(events[0,:,ordering.find("y")])-min(events[0,:,ordering.find("y")])+1
                        self.sensformat((int(size_x.item()),int(size_y.item())))
                        events[0,:,ordering.find("x")] -= min(events[0,:,ordering.find("x")]).numpy()
                        events[0,:,ordering.find("y")] -= min(events[0,:,ordering.find("y")]).numpy()
                        
                    for iev in range(N_max):
                        self.run(events[0][iev][ordering.find("x")].item(), \
                                 events[0][iev][ordering.find("y")].item(), \
                                 events[0][iev][ordering.find("t")].item(), \
                                 events[0][iev][ordering.find("p")].item(), \
                                 learn=True)
            pbar.close()
            for l in range(len(self.L)):
                self.stats[l].histo = self.L[l].cumhisto.copy()
                
            self.save_model(dataset)
            return self

    def run(self, x, y, t, p, learn=False):
        lay = 0
        activout=False
        while lay<len(self.TS):
            timesurf = self.TS[lay].addevent(x, y, t, p)
            if activ:
                p, dist = self.L[lay].run(timesurf, learn)
                if to_record:
                    self.stats[lay].update(p, self.L[lay].kernel, timesurf, dist)
                    #self.stats[lay].actmap[int(np.argmax(p)),self.TS[lay].x,self.TS[lay].y]=1
                lay+=1
                if lay==len(self.TS):
                    activout=True
            else:
                lay = len(self.TS)
        out = [x,y,t,np.argmax(p)]
        return out, activout

    def get_fname(self):
        timestr = self.date
        algo = self.L[0].algo
        arch = [self.L[i].kernel.shape[1] for i in range(len(self.L))]
        R = [self.L[i].R for i in range(len(self.L))]
        tau = [np.round(self.TS[i].tau*1e-3,2) for i in range(len(self.TS))]
        homeo = self.L[0].homeo
        homparam = self.L[0].homparam
        krnlinit = self.L[0].krnlinit
        sigma = self.TS[0].sigma
        onebyone = self.onbon
        f_name = f'{timestr}_{algo}_{krnlinit}_{sigma}_{homeo}_{homparam}_{arch}_{tau}_{R}_{onebyone}'
        self.name = f_name
        return f_name
    
    def sensformat(self,sensor_size):
        for i in range(1,len(self.TS)):
            self.TS[i].camsize = sensor_size
            self.TS[i].spatpmat = np.zeros((self.L[i-1].kernel.shape[1],sensor_size[0]+1,sensor_size[1]+1))
            self.stats[i].actmap = np.zeros((self.L[i-1].kernel.shape[1],sensor_size[0]+1,sensor_size[1]+1))
        self.TS[0].camsize = sensor_size
        self.TS[0].spatpmat = np.zeros((2,sensor_size[0]+1,sensor_size[1]+1))
        self.stats[0].actmap = np.zeros((2,sensor_size[0]+1,sensor_size[1]+1))

    def save_model(self, dataset):
        if dataset=='nmnist':
            path = '../Records/EXP_03_NMNIST/models/'
        elif dataset=='cars':
            path = '../Records/EXP_04_NCARS/models/'
        elif dataset=='poker':
            path = '../Records/EXP_05_POKERDVS/models/'
        elif dataset=='gesture':
            path = '../Records/EXP_06_DVSGESTURE/models/'
        else: print('define a path for this dataset')
        if not os.path.exists(path):
            os.makedirs(path)
        f_name = path+self.get_fname()+'.pkl'
        with open(f_name, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def load_model(self, dataset, verbose):
        model = []
        if dataset=='nmnist':
            path = '../Records/EXP_03_NMNIST/models/'
        elif dataset=='cars':
            path = '../Records/EXP_04_NCARS/models/'
        elif dataset=='poker':
            path = '../Records/EXP_05_POKERDVS/models/'
        elif dataset=='gesture':
            path = '../Records/EXP_06_DVSGESTURE/models/'
        else: print('define a path for this dataset')
        f_name = path+self.get_fname()+'.pkl'
        if verbose:
            print(f_name)
        if not os.path.isfile(f_name):
            return model
        else:
            with open(f_name, 'rb') as file:
                model = pickle.load(file)
        return model

    def save_output(self, evout, homeo, dataset, nb, train, jitonic, outstyle, kfold_ind):
        if dataset=='nmnist':
            direc = 'EXP_03_NMNIST'
        elif dataset=='cars':
            direc = 'EXP_04_NCARS'
        elif dataset=='poker':
            direc = 'EXP_05_POKERDVS'
        elif dataset=='gesture':
            direc = 'EXP_06_DVSGESTURE'
        elif dataset=='barrel':
            direc = 'EXP_01_LagorceKmeans'
        else: print('define a path for this dataset')
        if train:
            path = f'../Records/{direc}/train/'
        else:
            path = f'../Records/{direc}/test/'
        if not os.path.exists(path):
            os.makedirs(path)
        f_name = path+self.get_fname()+f'_{nb}_{jitonic}_{outstyle}'
        if kfold_ind is not None:
            f_name+='_'+str(kfold_ind)
        if homeo:
            f_name = f_name+'_homeo'
        f_name = f_name +'.pkl'
        with open(f_name, 'wb') as file:
            pickle.dump(evout, file, pickle.HIGHEST_PROTOCOL)

    def load_output(self, dataset, homeo, nb, train, jitonic, outstyle, kfold_ind, verbose):
        loaded = False
        output = []
        if dataset=='nmnist':
            direc = 'EXP_03_NMNIST'
        elif dataset=='cars':
            direc = 'EXP_04_NCARS'
        elif dataset=='poker':
            direc = 'EXP_05_POKERDVS'
        elif dataset=='gesture':
            direc = 'EXP_06_DVSGESTURE'
        else: print('define a path for this dataset')
        if train:
            path = f'../Records/{direc}/train/'
        else:
            path = f'../Records/{direc}/test/'
        f_name = path+self.get_fname()+f'_{nb}_{jitonic}_{outstyle}'
        if kfold_ind is not None:
            f_name+='_'+str(kfold_ind)
        if homeo:
            f_name = f_name+'_homeo'
        f_name = f_name +'.pkl'
        if verbose:
            print(f_name)
        if os.path.isfile(f_name):
            with open(f_name, 'rb') as file:
                output = pickle.load(file)
            loaded = True
        return output, loaded


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


def load(dataset, trainset, jitonic, kfold, kfold_ind):

    #_______ADDING JITTER_________
    transform = None
    if jitonic[1] is not None:
        print(f'spatial jitter -> var = {jitonic[1]}')
        transform = tonic.transforms.Compose([tonic.transforms.SpatialJitter(variance_x=jitonic[1], variance_y=jitonic[1], sigma_x_y=0, integer_coordinates=True, clip_outliers=True)])

    if jitonic[0] is not None:
        print(f'time jitter -> var = {jitonic[0]}')
        transform = tonic.transforms.Compose([tonic.transforms.TimeJitter(variance=jitonic[0], integer_timestamps=False, clip_negative=True, sort_timestamps=True)])
    #_____________________________
         
    #_______GETTING DATASET_______    
    path = '../Data/'
    if dataset == 'nmnist':
        eventset = tonic.datasets.NMNIST(save_to='../Data/',
                                train=trainset,
                                transform=transform)
    elif dataset == 'poker':
        eventset = tonic.datasets.POKERDVS(save_to='../Data/',
                                train=trainset,
                                transform=transform)
    elif dataset == 'gesture':
        eventset = tonic.datasets.DVSGesture(save_to='../Data/',
                                train=trainset,
                                transform=transform)
    elif dataset == 'cars':
        eventset = tonic.datasets.NCARS(save_to='../Data/',
                                train=trainset,
                                transform=transform)
    elif dataset == 'ncaltech':
        eventset = tonic.datasets.NCALTECH101(save_to='../Data/',
                                train=trainset,
                                transform=transform)
    else: print('problem with dataset')
    #_____________________________
        
        
    #_______BUILDING LOADER_______  
    if kfold:
        subset_indices = []
        subset_size = len(eventset)//kfold
        for i in range(len(eventset.classes)):
            all_ind = np.where(np.array(eventset.targets)==i)[0]
            subset_indices += all_ind[kfold_ind*subset_size//len(eventset.classes):
                            min((kfold_ind+1)*subset_size//len(eventset.classes), len(eventset)-1)].tolist()
        g_cpu = Generator()
        subsampler = SubsetRandomSampler(subset_indices, g_cpu)
        loader = DataLoader(eventset, batch_size=1, shuffle=False, sampler=subsampler)
    else:
        loader = DataLoader(eventset, shuffle=True)
    #_____________________________
    return loader, eventset.ordering, eventset.classes
    