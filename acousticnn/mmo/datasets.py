import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from acousticModels import MultiMassOscillator


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


fixed_values = {"n_masses": 2, 
                "sample_f": True,
                "f_range": (-1.5, 1), 
                "f_per_sample": 10,
                "u": 1, 
                "sample_m": False,
                "m_range": (1, 125),
                "sample_d": False,
                "d_range": (0.1, 1),
                "sample_k": True,
                "k_range": (1, 125),
                "normalize": True,
                "normalize_factor": 100
                }


class Iter_Dataset(Dataset):
    def __init__(self, n_samples=100, parameters=None):
        super().__init__()
        fixed_values.update(parameters)
        self.parameters = AttrDict(fixed_values)
        print(parameters)
        self.n_samples = n_samples
        self.init_fixed_parameters()
        self.generate_samples()
        self.calls_per_worker = {}
        
    def init_fixed_parameters(self):
        self.m_fixed = np.ones((self.n_samples, self.parameters.n_masses))
        self.d_fixed = np.ones((self.n_samples, self.parameters.n_masses+1))
        self.k_fixed = np.ones((self.n_samples, self.parameters.n_masses+1))
        self.list_of_couplings = []
        self.f_fixed = np.logspace(*self.parameters.f_range, self.parameters.f_per_sample)  # 200
        self.f_fixed = np.tile(self.f_fixed, (self.n_samples, 1))  # n_samples x 200
        self.generation_counter = 0 

    def generate_samples(self):
        self.amplitude = []
        
        # parameters of oscillator
        if self.parameters.sample_m is True:
            print(*self.parameters.m_range)
            self.m_sample = np.random.uniform(*self.parameters.m_range, (self.n_samples, self.parameters.n_masses))
        else:
            self.m_sample = self.m_fixed
        if self.parameters.sample_d is True:
            print(*self.parameters.d_range)
            self.d_sample = np.random.uniform(*self.parameters.d_range, (self.n_samples, self.parameters.n_masses+1))
        else:
            self.d_sample = self.d_fixed
        if self.parameters.sample_k is True:
            print(*self.parameters.k_range)
            self.k_sample = np.random.uniform(*self.parameters.k_range, (self.n_samples, self.parameters.n_masses+1))
        else:
            self.k_sample = self.k_fixed
        if self.parameters.sample_f is True:
            self.f_sample = 10**np.random.uniform(*self.parameters.f_range, (self.n_samples, self.parameters.f_per_sample))
            print("sample_f")
        else:
            self.f_sample = self.f_fixed
            print("fix_f")

        # calculate frequency response
        for f, m, d, k in zip(np.array(self.f_sample), self.m_sample, self.d_sample, self.k_sample):                
            oscillator = MultiMassOscillator(m, d, k, self.parameters.u, self.list_of_couplings)
            self.amplitude.append(oscillator.frequency_response(f)) # 1 x 200, 4
        self.amplitude = np.log10(np.abs(np.array(self.amplitude)))[:,0] # n_samples x 200 x 4
        self.not_transformed_amplitude = np.array(self.amplitude).copy()
        self.f_sample = torch.from_numpy(self.f_sample).float()
        self.amplitude = torch.from_numpy(self.amplitude).float() 
        self.m_sample = torch.from_numpy(self.m_sample).float()
        self.d_sample = torch.from_numpy(self.d_sample).float()
        self.k_sample = torch.from_numpy(self.k_sample).float()
        if self.parameters.normalize:
            self.normalize_samples()
        self.generation_counter += 1

    def normalize_samples(self):
        print("Normalize")
        print(self.parameters.normalize)
        print(torch.mean(self.f_sample), torch.std(self.f_sample))
        print(torch.mean(self.m_sample), torch.std(self.m_sample))
        print(torch.mean(self.d_sample), torch.std(self.d_sample))
        print(torch.mean(self.k_sample), torch.std(self.k_sample))
        self.f_sample = (self.f_sample - 10**np.mean(np.array(self.parameters.f_range)) -1) / 10**np.array(self.parameters.f_range[1])*self.parameters.normalize_factor
        self.m_sample = (self.m_sample - np.mean(self.parameters.m_range)) / self.parameters.m_range[1]*self.parameters.normalize_factor
        self.d_sample = (self.d_sample - np.mean(self.parameters.d_range)) / self.parameters.d_range[1]*self.parameters.normalize_factor
        self.k_sample = (self.k_sample - np.mean(self.parameters.k_range)) / self.parameters.k_range[1]*self.parameters.normalize_factor
        print(torch.mean(self.f_sample), torch.std(self.f_sample))
        print(torch.mean(self.m_sample), torch.std(self.m_sample))
        print(torch.mean(self.d_sample), torch.std(self.d_sample))
        print(torch.mean(self.k_sample), torch.std(self.k_sample))

    def __getitem__(self, idx):
        # returns frequency, (parameters), response
        # [200], ([4], [5], [5]), [200, 4]
        return self.f_sample[idx], (self.m_sample[idx], self.d_sample[idx], self.k_sample[idx]), self.amplitude[idx]

    def __len__(self):
        return self.n_samples


def get_dataloader(args, config, test=False, n_samples=128, parameters={}, n_workers=None, batch_size=10):
    if test is True:
        shuffle = False
        drop_last = False
        num_workers = 0
        parameters["sample_f"] = False
    else:
        shuffle = True
        drop_last = True
        num_workers = 0
    if n_workers is not None:
        num_workers = n_workers
    dataset = Iter_Dataset(n_samples=n_samples, parameters=parameters)
    return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last,shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    torch.manual_seed(np.random.get_state()[1][0] + worker_id)


def get_listdataloader(args, config, parameters_list, n_samples, batch_size):
    shuffle = True
    drop_last = True
    num_workers = 0
    dataset = ListDataset(parameters_list, n_samples, batch_size)
    return DataLoader(dataset, batch_size=1, drop_last=drop_last, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn, collate_fn=col)

def col(batch):
    return batch[0]


class ListDataset(Dataset):
    def __init__(self, parameters_list, n_samples, batch_size):
        self.dataloaders = []
        self.n_samples = n_samples
        self.n_dataloaders = len(parameters_list)
        self.batch_size = batch_size
        for group in parameters_list:
            parameters = parameters_list[group]
            self.dataloaders.append(get_dataloader(None, None, test=False, n_samples=n_samples, parameters=parameters, batch_size=batch_size))

    def __getitem__(self, idx):
        i = int(np.floor(idx / len(self) * self.n_dataloaders))
        return next(iter(self.dataloaders[int(i)]))

    def __len__(self):
        return np.sum(len(dl) for dl in self.dataloaders)

