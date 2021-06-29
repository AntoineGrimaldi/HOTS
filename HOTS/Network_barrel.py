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
from torch.utils.data import SubsetRandomSampler

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

    def __init__(self,  timestr = None,
                        # architecture of the network (default=Lagorce2017)
                        nbclust = [4, 8, 16],
                        # parameters of time-surfaces and datasets
                        tau = 10, #timestamp en millisec/
                        K_tau = 10,
                        decay = 'exponential', # among ['exponential', 'linear']
                        nbpolcam = 2,
                        R = 2,
                        K_R = 2,
                        camsize = (34, 34),
                        # functional parameters of the network
                        algo = 'lagorce', # among ['lagorce', 'maro', 'mpursuit']
                        krnlinit = 'rdn',
                        hout = False, #works only with mpursuit
                        homeo = False,
                        homparam = [.25, 1],
                        pola = True,
                        to_record = True,
                        filt = 2,
                        sigma = None,
                        jitter = False,
                        homeinv = False,
                ):
        self.jitter = jitter # != from jitonic, this jitter is added at the layer output, creating an average pooling
        self.onbon = False
        self.name = 'hots'
        self.date = timestr
        tau *= 1e3 # to enter tau in ms
        nblay = len(nbclust)
        if to_record:
            self.stats = [[]]*nblay
        self.TS = [[]]*nblay
        self.L = [[]]*nblay
        for lay in range(nblay):
            if lay == 0:
                self.TS[lay] = TimeSurface(R, tau, camsize, nbpolcam, pola, filt, sigma)
                self.L[lay] = layer(R, nbclust[lay], pola, nbpolcam, homeo, homparam, homeinv, algo, hout, krnlinit, to_record)
                if to_record:
                    self.stats[lay] = stats(nbclust[lay], camsize)
            else:
                self.TS[lay] = TimeSurface(R*(K_R**lay), tau*(K_tau**lay), camsize, nbclust[lay-1], pola, filt, sigma)
                #self.L[lay] = layer(R*(K_R**lay), nbclust*(K_clust**lay), pola, nbclust*(K_clust**(lay-1)), homeo, homparam, homeinv, algo, hout, krnlinit, to_record)
                self.L[lay] = layer(R*(K_R**lay), nbclust[lay], pola, nbclust[lay-1], homeo, homparam, homeinv, algo, hout, krnlinit, to_record)
                if to_record:
                    self.stats[lay] = stats(nbclust[lay], camsize)  


##___________REPRODUCING RESULTS FROM LAGORCE 2017___________________________________________
##___________________________________________________________________________________________

    def learninglagorce(self, nb_cycle=3, diginit=True, filtering=None):


        #___________ SPECIAL CASE OF SIMPLE_ALPHABET DATASET _________________

        path = "../Data/alphabet_ExtractedStabilized.mat"

        image_list = [1, 32, 19, 22, 29]
        for i in range(nb_cycle-1):
            image_list += image_list
        address, time, polarity, list_pola = LoadFromMat(path, image_number=image_list)

        #___________ SPECIAL CASE OF SIMPLE_ALPHABET DATASET _________________

        nbevent = int(time.shape[0])
        for n in range(len(self.L)):
            count = 0
            pbar = tqdm(total=nbevent)
            while count<nbevent:
                pbar.update(1)
                x,y,t,p = address[count,0],address[count,1], time[count],polarity[count]
                if diginit and time[count]<time[count-1]:
                    for i in range(n+1):
                        self.TS[i].spatpmat[:] = 0
                        self.TS[i].iev = 0
                lay=0
                while lay < n+1:
                    if lay==n:
                        learn=True
                    else:
                        learn=False
                    timesurf, activ = self.TS[lay].addevent(x, y, t, p)
                    if lay==0 or filtering=='all':
                        activ2=activ
                    if activ2 and np.sum(timesurf)>0:
                        p, dist = self.L[lay].run(timesurf, learn)
                        if learn:
                            self.stats[lay].update(p, self.L[lay].kernel, timesurf, dist)
                        lay += 1
                    else:
                        lay = n+1
                count += 1
            for l in range(len(self.L)):
                self.stats[l].histo = self.L[l].cumhisto.copy()
            pbar.close()

    def traininglagorce(self, nb_digit=None, outstyle = 'histo', to_record=True):
        
        class_data = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "U": 20,
    "V": 21,
    "W": 22,
    "X": 23,
    "Y": 24,
    "Z": 25,
    "0": 26,
    "1": 27,
    "2": 28,
    "3": 29,
    "4": 30,
    "5": 31,
    "6": 32,
    "7": 33,
    "8": 34,
    "9": 35,
}
        
        path = "../Data/alphabet_ExtractedStabilized.mat"
        nblist = 36
        image_list=list(np.arange(0, nblist))
        address, time, polarity, list_pola = LoadFromMat(path, image_number=image_list)
        with open('../Data/alphabet_label.pkl', 'rb') as file:
            label_list = pickle.load(file)
        label = label_list[:nblist]

        learn=False
        output = []
        count = 0
        count2 = 0
        nbevent = int(time.shape[0])
        pbar = tqdm(total=nbevent)
        idx = 0
        labelmap = []
        timout = []
        xout = []
        yout = []
        polout = []
        labout = []
        for i in range(len(self.L)):
            self.TS[i].spatpmat[:] = 0
            self.TS[i].iev = 0
            self.L[i].cumhisto[:] = 1

        while count<nbevent:
            pbar.update(1)
            out, activout = self.run(address[count,0],address[count,1],time[count],polarity[count], learn, to_record)
            if outstyle=='LR' and activout:
                xout.append(out[0])
                yout.append(out[1])
                timout.append(out[2])
                polout.append(out[3])
                labout.append(class_data[label[idx][0]])
                
            if count2==label[idx][1]:
                data = (label[idx][0],self.L[-1].cumhisto.copy())
                labelmap.append(data)
                for i in range(len(self.L)):
                    self.TS[i].spatpmat[:] = 0
                    self.TS[i].iev = 0
                    self.L[i].cumhisto[:] = 1
                idx += 1
                count2=-1
            count += 1
            count2 += 1
        pbar.close()
        if outstyle=='LR':
            camsize = self.TS[-1].camsize
            nbpola = self.L[-1].kernel.shape[1]
            eventsout = [xout,yout,timout,polout,labout,camsize,nbpola]
            self.date = '2020-12-01'
            self.save_output(eventsout, False, 'barrel', len(label), True, None, 'LR', None)
        return labelmap

    def testinglagorce(self, nb_digit=None, outstyle = 'histo', to_record=True):
        
        class_data = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "U": 20,
    "V": 21,
    "W": 22,
    "X": 23,
    "Y": 24,
    "Z": 25,
    "0": 26,
    "1": 27,
    "2": 28,
    "3": 29,
    "4": 30,
    "5": 31,
    "6": 32,
    "7": 33,
    "8": 34,
    "9": 35,
}
        
        path = "../Data/alphabet_ExtractedStabilized.mat"
        image_list=list(np.arange(36, 76))
        address, time, polarity, list_pola = LoadFromMat(path, image_number=image_list)
        with open('../Data/alphabet_label.pkl', 'rb') as file:
            label_list = pickle.load(file)
        label = label_list[36:76]

        learn = False
        output = []
        count = 0
        count2 = 0
        nbevent = int(time.shape[0])
        pbar = tqdm(total=nbevent)
        idx = 0
        labelmap = []
        timout = []
        xout = []
        yout = []
        polout = []
        labout = []
        for i in range(len(self.L)):
            self.TS[i].spatpmat[:] = 0
            self.TS[i].iev = 0
            self.L[i].cumhisto[:] = 1
        while count<nbevent:
            pbar.update(1)
            out, activout = self.run(address[count,0],address[count,1],time[count],polarity[count], learn, to_record)
            if outstyle=='LR' and activout:
                xout.append(out[0])
                yout.append(out[1])
                timout.append(out[2])
                polout.append(out[3])
                labout.append(class_data[label[idx][0]])
            if count2==label[idx][1]:
                data = (label[idx][0],self.L[-1].cumhisto.copy())
                labelmap.append(data)
                for i in range(len(self.L)):
                    self.TS[i].spatpmat[:] = 0
                    self.TS[i].iev = 0
                    self.L[i].cumhisto[:] = 1
                idx += 1
                count2=-1
            count += 1
            count2 += 1

        pbar.close()
        if outstyle=='LR':
            camsize = self.TS[-1].camsize
            nbpola = self.L[-1].kernel.shape[1]
            eventsout = [xout,yout,timout,polout,labout,camsize,nbpola]
            self.date = '2020-12-01'
            self.save_output(eventsout, False, 'barrel', len(label), False, None, 'LR', None)

        return labelmap

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
            plt.bar(np.arange(N[i]), self.stats[i].histo/np.sum(self.stats[i].histo), width=1, align='edge', ec="k")
            ax.set_xticks(())
            #if i>0:
                #ax.set_yticks(())
            ax.set_title('Layer '+str(i+1), fontsize=16)
            plt.xlim([0,N[i]])
            yhis = 1.1*max(self.stats[i].histo/np.sum(self.stats[i].histo))
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

#__________________OLD_CODE___________________________________________________________________________
#_____________________________________________________________________________________________________

def LoadFromMat(path, image_number, OutOnePolarity=False, verbose=0):
    '''
            Load Events from a .mat file. Only the events contained in ListPolarities are kept:
            INPUT
                + path : a string which is the path of the .mat file (ex : './data_cache/alphabet_ExtractedStabilized.mat')
                + image_number : list with all the numbers of image to load
    '''
    from scipy import io
    obj = io.loadmat(path)
    ROI = obj['ROI'][0]

    if type(image_number) is int:
        image_number = [image_number]
    elif type(image_number) is not list:
        raise TypeError(
                    'the type of argument image_number should be int or list')
    if verbose > 0:
        print("loading images {0}".format(image_number))
    Total_size = 0
    for idx, each_image in enumerate(image_number):
        image = ROI[each_image][0, 0]
        Total_size += image[1].shape[1]

    address = np.zeros((Total_size, 2)).astype(int)
    time = np.zeros((Total_size))
    polarity = np.zeros((Total_size))
    first_idx = 0

    for idx, each_image in enumerate(image_number):
        image = ROI[each_image][0, 0]
        last_idx = first_idx + image[0].shape[1]
        address[first_idx:last_idx, 0] = (image[1] - 1).astype(int)
        address[first_idx:last_idx, 1] = (image[0] - 1).astype(int)
        time[first_idx:last_idx] = (image[3] * 1e-6)
        polarity[first_idx:last_idx] = image[2].astype(int)
        first_idx = last_idx

    polarity[polarity.T == -1] = 0
    polarity = polarity.astype(int)
            # Filter only the wanted polarity
    ListPolarities = np.unique(polarity)
    if OutOnePolarity == True:
        polarity = np.zeros_like(polarity)
        ListPolarities = [0]

    return address, time, polarity, ListPolarities
