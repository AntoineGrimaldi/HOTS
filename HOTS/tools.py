from torch.utils.data import SubsetRandomSampler, DataLoader
from torch import Generator
from tonic.dataset import Dataset
from network import network
import numpy as np
import os



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
        g_cpu = Generator()
        g_cpu.manual_seed(seed)
        subsampler = SubsetRandomSampler(subset_indices, g_cpu)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=subsampler, num_workers = num_workers)
    else:
        loader = DataLoader(dataset, shuffle=True, num_workers = num_workers)
    return loader, dataset.ordering, dataset.classes



class HOTS_Dataset(Dataset):
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
        #events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
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