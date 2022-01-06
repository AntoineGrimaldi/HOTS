from network import network
import numpy as np
import os, torch, tonic, pickle
from tqdm import tqdm

def get_loader(dataset, kfold = None, kfold_ind = 0, num_workers = 0, seed=42):
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
        loader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers = num_workers)
    return loader

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
    
    path_to_dataset = f'../Records/output/train/{network.get_fname()}_None/'
    if not os.path.exists(path_to_dataset):
        print('process samples with the HOTS network first')
        return
    timesurface_size = (network.TS[0].camsize[0], network.TS[0].camsize[1], network.L[-1].kernel.shape[1])
    
    transform = tonic.transforms.Compose([tonic.transforms.ToTimesurface(sensor_size=timesurface_size, tau=tau_cla, decay="exp")])
    dataset = HOTS_Dataset(path_to_dataset, timesurface_size, transform=transform)
    loader = get_loader(dataset, kfold = kfold, kfold_ind = kfold_ind, num_workers = num_workers, seed=seed)
    if verbose: print(f'Number of training samples: {len(loader)}')
    model_name = f'../Records/models/{network.get_fname()}_{len(loader)}_LR.pkl'
    
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
            pbar.update(1)

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
                verbose=False,
        ):
    
    path_to_dataset = f'../Records/output/test/{network.get_fname()}_{jitter}/'
    if not os.path.exists(path_to_dataset):
        print('process samples with the HOTS network first')
        return
    timesurface_size = (network.TS[0].camsize[0], network.TS[0].camsize[1], network.L[-1].kernel.shape[1])
    N = timesurface_size[0]*timesurface_size[1]*timesurface_size[2]
    
    transform = tonic.transforms.Compose([tonic.transforms.ToTimesurface(sensor_size=timesurface_size, tau=tau_cla, decay="exp")])
    dataset = HOTS_Dataset(path_to_dataset, timesurface_size, transform=transform)
    loader = get_loader(dataset, kfold = kfold, kfold_ind = kfold_ind, num_workers = num_workers, seed=seed)
    if verbose: print(f'Number of testing samples: {len(loader)}')
    
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f'device -> {device} - num workers -> {num_workers}')

        logistic_model = model.to(device)

        if verbose:
            pbar = tqdm(total=len(loader))
        likelihood, true_target = [], []

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

    return likelihood, true_target

class LRtorch(torch.nn.Module):
    #torch.nn.Module -> Base class for all neural network modules
    def __init__(self, N, n_classes, bias=True):
        super(LRtorch, self).__init__()
        self.linear = torch.nn.Linear(N, n_classes, bias=bias)
        self.nl = torch.nn.Softmax(dim=1)

    def forward(self, factors):
        return self.nl(self.linear(factors))