from network import network
import numpy as np
import os, torch, tonic, pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

## DATASET

def get_loader(dataset, kfold = None, kfold_ind = 0, num_workers = 0, shuffle=True, seed=42):
    # creates a loader for the samples of the dataset. If kfold is not None, 
    # then the dataset is splitted into different folds with equal repartition of the classes.
    if kfold:
        subset_indices = []
        subset_size = len(dataset)//kfold
        for i in range(len(dataset.classes)):
            all_ind = np.where(np.array(dataset.targets)==i)[0]
            subset_indices += all_ind[kfold_ind*subset_size//len(dataset.classes):
                            min((kfold_ind+1)*subset_size//len(dataset.classes), len(dataset)-1)].tolist()
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        subsampler = torch.utils.data.SubsetRandomSampler(subset_indices, g_cpu)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=subsampler, num_workers = num_workers)
    else:
        loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, num_workers = num_workers)
    return loader

def get_isi(events, ordering = 'xytp', verbose = False):
    t_index, p_index = ordering.index('t'), ordering.index('p')
    mean_isi = None
    isipol = np.zeros([2])
    for polarity in [0,1]:
        events_pol = events[(events[:, p_index]==polarity)]
        N_events = events_pol.shape[0]-1
        for i in range(events_pol.shape[0]-1):
            isi = events_pol[i+1,t_index]-events_pol[i,t_index]
            if isi>0:
                mean_isi = (N_events-1)/N_events*mean_isi+1/N_events*isi if mean_isi else isi
        isipol[polarity]=mean_isi
    if verbose:
        print(f'Mean ISI for ON events: {np.round(isipol[1].mean()*1e-3,1)} in ms \n')
        print(f'Mean ISI for OFF events: {np.round(isipol[0].mean()*1e-3,1)} in ms \n')
    return isipol

def get_dataset_info(trainset, testset):
    t_index = trainset.ordering.index("t")
    
    print(f'number of samples in the trainset: {len(trainset)}')
    print(f'number of samples in the testset: {len(testset)}')
    print(40*'-')
    
    nbev = []
    recordingtime = []
    mean_isi = []
    
    loader = get_loader(trainset)
    for events, target in loader:
        events = events.squeeze().numpy()
        mean_isi.append(get_isi(events,trainset.ordering).mean())
        nbev.append(len(events))
        recordingtime.append(events[:,t_index][-1])
    loader = get_loader(testset)
    for events, target in loader:
        events = events.squeeze().numpy()
        mean_isi.append(get_isi(events,trainset.ordering).mean())
        nbev.append(len(events))
        recordingtime.append(events[:,t_index][-1])
        
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    for i in range(3):
        if i == 0:
            x = recordingtime
            ttl = 'recording time (in $\mu s$)'
        elif i == 1:
            x = nbev
            ttl = 'number of events '
        else:
            x = mean_isi
            ttl = 'mean ISI (in $\mu s$)'

        n, bins, patches = axs[i].hist(x=x, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        axs[i].grid(axis='y', alpha=0.75)
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(f'Histogram for the {ttl}')
        maxfreq = n.max()
        axs[i].set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    print(f'mean value for the recording time: {np.round(np.mean(recordingtime),0)/1e3} ms')
    print(f'mean value for the number of events: {int(np.round(np.mean(nbev),0))}')
    print(f'mean value for the interspike interval: {int(np.round(np.nanmean(mean_isi),0))} us')
    print(40*'-')
    

class HOTS_Dataset(tonic.dataset.Dataset):
    """Make a dataset from the output of the HOTS network
    """
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, path_to, sensor_size, train=True, transform=None, target_transform=None):
        super(HOTS_Dataset, self).__init__(
            path_to, transform=transform, target_transform=target_transform
        )

        self.location_on_system = path_to
        
        if not os.path.exists(self.location_on_system):
            print('no output, process the samples first')
            return

        self.sensor_size = sensor_size
        
        for path, dirs, files in os.walk(self.location_on_system):
            files.sort()
            if dirs:
                label_length = len(dirs[0])
                self.classes = dirs
                self.int_classes = dict(zip(self.classes, range(len(dirs))))
            for file in files:
                if file.endswith("npy"):
                    self.data.append(np.load(os.path.join(path, file)))
                    self.targets.append(self.int_classes[path[-label_length:]])

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events, target = self.data[index], self.targets[index]
        events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._is_file_present() and self._folder_contains_at_least_n_files_of_type(
            20, ".npy"
        )
    
## MLR
    
def fit_MLR(network, 
            tau_cla, #enter tau_cla in ms
            kfold = None,
            kfold_ind = 0,
        #parameters of the model learning
            num_workers = 0, # ajouter num_workers si besoin!
            learning_rate = 0.005,
            betas = (0.9, 0.999),
            num_epochs = 2 ** 5 + 1,
            seed = 42,
            verbose=True):
    
    tau_cla*=1e3
    path_to_dataset = f'../Records/output/train/{network.get_fname()}_None/'
    if not os.path.exists(path_to_dataset):
        print('process samples with the HOTS network first')
        return
    timesurface_size = (network.TS[0].camsize[0], network.TS[0].camsize[1], network.L[-1].kernel.shape[1])
    
    transform = tonic.transforms.Compose([tonic.transforms.ToTimesurface(sensor_size=timesurface_size, tau=tau_cla, decay="exp")])
    dataset = HOTS_Dataset(path_to_dataset, timesurface_size, transform=transform)
    loader = get_loader(dataset, kfold = kfold, kfold_ind = kfold_ind, num_workers = num_workers, seed=seed)
    if verbose: print(f'Number of training samples: {len(loader)}')
    model_name = f'../Records/models/{network.get_fname()}_{int(tau_cla)}_{len(loader)}_LR.pkl'
    
    if os.path.isfile(model_name):
        print('load existing model')
        with open(model_name, 'rb') as file:
            logistic_model, losses = pickle.load(file)
    else:
        torch.set_default_tensor_type("torch.DoubleTensor")
        criterion = torch.nn.BCELoss(reduction="mean")
        amsgrad = True #or False gives similar results
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose: print(f'device -> {device} - num workers -> {num_workers}')

        N = timesurface_size[0]*timesurface_size[1]*timesurface_size[2]
        n_classes = len(dataset.classes)
        logistic_model = LRtorch(N, n_classes)
        logistic_model = logistic_model.to(device)
        logistic_model.train()
        optimizer = torch.optim.Adam(
            logistic_model.parameters(), lr=learning_rate, betas=betas, amsgrad=amsgrad
        )
        if not verbose:
            pbar = tqdm(total=int(num_epochs))
        for epoch in range(int(num_epochs)):
            losses = []
            for X, label in loader:
                X, label = X.to(device), label.to(device)
                X, label = X.squeeze(0), label.squeeze(0) # just one digit = one batch
                X = X.reshape(X.shape[0], N)

                outputs = logistic_model(X)

                n_events = X.shape[0]
                labels = label*torch.ones(n_events).type(torch.LongTensor).to(device)
                labels = torch.nn.functional.one_hot(labels, num_classes=n_classes).type(torch.DoubleTensor).to(device)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if verbose:
                print(f'loss for epoch number {epoch}: {loss}')
            else:
                pbar.update(1)
        if not verbose:
            pbar.close()
        with open(model_name, 'wb') as file:
            pickle.dump([logistic_model, losses], file, pickle.HIGHEST_PROTOCOL)

    return logistic_model, losses

def predict_MLR(network, 
                model,
                tau_cla, #enter tau_cla in ms
                jitter = None,
                patch_size = None,
                kfold = None,
                kfold_ind = 0,
                num_workers = 0,
                seed=42,
                verbose=True,
        ):
    
    tau_cla*=1e3
    path_to_dataset = f'../Records/output/test/{network.get_fname()}_{jitter}/'
    if not os.path.exists(path_to_dataset):
        print('process samples with the HOTS network first')
        return
    timesurface_size = (network.TS[0].camsize[0], network.TS[0].camsize[1], network.L[-1].kernel.shape[1])
    N = timesurface_size[0]*timesurface_size[1]*timesurface_size[2]
    
    shuffle=False
    transform = tonic.transforms.Compose([tonic.transforms.ToTimesurface(sensor_size=timesurface_size, tau=tau_cla, decay="exp")])
    dataset = HOTS_Dataset(path_to_dataset, timesurface_size, transform=transform)
    loader = get_loader(dataset, kfold = kfold, kfold_ind = kfold_ind, num_workers = num_workers, shuffle=shuffle, seed=seed)
    
    dataset_for_timestamps = HOTS_Dataset(path_to_dataset, timesurface_size, transform=tonic.transforms.NumpyAsType(int))
    loader_for_timestamps = get_loader(dataset_for_timestamps, kfold = kfold, kfold_ind = kfold_ind, num_workers = num_workers, shuffle=shuffle, seed=seed)
    if verbose: print(f'Number of testing samples: {len(loader)}')
    
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f'device -> {device} - num workers -> {num_workers}')

        logistic_model = model.to(device)

        if verbose:
            pbar = tqdm(total=len(loader))
        likelihood, true_target, timestamps = [], [], []

        for X, label in loader:
            X, label = X[0].to(device) ,label[0].to(device)
            X = X.reshape(X.shape[0], N)
            n_events = X.shape[0]
            outputs = logistic_model(X)
            likelihood.append(outputs.cpu().numpy())
            true_target.append(label.cpu().numpy())
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()
            
        t_index = dataset_for_timestamps.ordering.index('t')
        for events, target in loader_for_timestamps:
            timestamps.append(events[0,:,t_index])

    return likelihood, true_target, timestamps

class LRtorch(torch.nn.Module):
    #torch.nn.Module -> Base class for all neural network modules
    def __init__(self, N, n_classes, bias=True):
        super(LRtorch, self).__init__()
        self.linear = torch.nn.Linear(N, n_classes, bias=bias)
        self.nl = torch.nn.Softmax(dim=1)

    def forward(self, factors):
        return self.nl(self.linear(factors))
    
def score_classif_events(likelihood, true_target, thres=None, verbose=True):
    
    max_len = 0
    for likeli in likelihood:
        if max_len<likeli.shape[0]:
            max_len=likeli.shape[0]

    matscor = np.zeros([len(true_target),max_len])
    matscor[:] = np.nan
    sample = 0
    lastac = 0
    nb_test = len(true_target)

    for likelihood_, true_target_ in zip(likelihood, true_target):
        pred_target = np.zeros(len(likelihood_))
        pred_target[:] = np.nan
        if not thres:
            pred_target = np.argmax(likelihood_, axis = 1)
        else:
            for i in range(len(likelihood_)):
                if np.max(likelihood_[i])>thres:
                    pred_target[i] = np.argmax(likelihood_[i])
        for event in range(len(pred_target)):
            if np.isnan(pred_target[event])==False:
                matscor[sample,event] = pred_target[event]==true_target_
        if pred_target[-1]==true_target_:
            lastac+=1
        sample+=1

    meanac = np.nanmean(matscor)
    onlinac = np.nanmean(matscor, axis=0)
    lastac/=nb_test
    truepos = len(np.where(matscor==1)[0])
    falsepos = len(np.where(matscor==0)[0])

    if verbose:
        print(f'Mean accuracy: {np.round(meanac,3)*100}%')
        plt.plot(onlinac);
        plt.xlabel('number of events');
        plt.ylabel('online accuracy');
        plt.title('LR classification results evolution as a function of the number of events');
    
    return meanac, onlinac, lastac, truepos, falsepos

def score_classif_time(likelihood, true_target, timestamps, timestep, thres=None, verbose=True):
    
    max_dur = 0
    for time in timestamps:
        if max_dur<time[-1]:
            max_dur=time[-1]
            
    time_axis = np.arange(0,max_dur,timestep)

    matscor = np.zeros([len(true_target),len(time_axis)])
    matscor[:] = np.nan
    sample = 0
    lastac = 0
    nb_test = len(true_target)
    
    for likelihood_, true_target_, timestamps_ in zip(likelihood, true_target, timestamps):
        pred_timestep = np.zeros(len(time_axis))
        pred_timestep[:] = np.nan
        for step in range(1,len(pred_timestep)):
            indices = np.where((timestamps_.numpy()<=time_axis[step])&(timestamps_.numpy()>time_axis[step-1]))[0]
            mean_likelihood = np.mean(likelihood_[indices,:],axis=0)
            if np.isnan(mean_likelihood).sum()>0:
                if not np.isnan(np.array(pred_timestep[step-1])):
                    pred_timestep[step] = pred_timestep[step-1]
                    #pred_timestep[step] = np.nan
            else:
                if not thres:
                    pred_timestep[step] = np.nanargmax(mean_likelihood)
                elif np.max(likelihood_[indices,np.nanargmax(mean_likelihood)])>thres:
                    pred_timestep[step] = np.nanargmax(mean_likelihood)
                elif not np.isnan(np.array(pred_timestep[step-1])):
                    pred_timestep[step] = pred_timestep[step-1]
                    #pred_timestep[step] = np.nan
            if not np.isnan(pred_timestep[step]):
                matscor[sample,step] = pred_timestep[step]==true_target_
        if pred_timestep[-1]==true_target_:
            lastac+=1
        sample+=1
        
    meanac = np.nanmean(matscor)
    onlinac = np.nanmean(matscor, axis=0)
    lastac/=nb_test
    truepos = len(np.where(matscor==1)[0])
    falsepos = len(np.where(matscor==0)[0])
        
    if verbose:
        print(f'Mean accuracy: {np.round(meanac,3)*100}%')
        plt.plot(time_axis*1e-3,onlinac);
        plt.xlabel('time (in ms)');
        plt.ylabel('online accuracy');
        plt.title('LR classification results evolution as a function of time');
    
    return meanac, onlinac, lastac, truepos, falsepos


## OTHER 
# classif avec histogram

def fit_histo(network, 
              num_workers=0,
              verbose=True):
    
    path_to_dataset = f'../Records/output/train/{network.get_fname()}_None/'
    if not os.path.exists(path_to_dataset):
        print('process samples with the HOTS network first')
        return
    
    timesurface_size = (network.TS[0].camsize[0], network.TS[0].camsize[1], network.L[-1].kernel.shape[1])
    dataset = HOTS_Dataset(path_to_dataset, timesurface_size, transform=tonic.transforms.NumpyAsType(int))
    loader = get_loader(dataset, num_workers = num_workers)
    if verbose: print(f'Number of training samples: {len(loader)}')
    model_name = f'../Records/models/{network.get_fname()}_{len(loader)}_histo.pkl'
    
    if os.path.isfile(model_name):
        print('load existing histograms')
        with open(model_name, 'rb') as file:
            histo, labelz = pickle.load(file)
    else:
        p_index = dataset.ordering.index('p')
        #n_classes = len(dataset.classes)
        n_polarity = timesurface_size[2]
        histo = np.zeros([len(loader),n_polarity])
        labelz = []
        pbar = tqdm(total=len(loader))
        sample_number = 0
        for events, label in loader:
            events, label = events.squeeze(0), label.squeeze(0) # just one digit = one batch
            labelz.append(label)
            value, frequency = np.unique(events[:,p_index], return_counts=True)
            histo[sample_number,[value]] = frequency
            sample_number+=1
        pbar.update(1)
        pbar.close()
        with open(model_name, 'wb') as file:
            pickle.dump([histo, labelz], file, pickle.HIGHEST_PROTOCOL)

    return histo, labelz