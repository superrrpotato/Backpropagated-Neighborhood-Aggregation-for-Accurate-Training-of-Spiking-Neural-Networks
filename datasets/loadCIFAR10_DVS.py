import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from torch.utils.data.sampler import SubsetRandomSampler
import math


class CIFAR10_DVS(Dataset):
    def __init__(self, dataset_path, n_steps):
        self.path = dataset_path
        self.n_steps = n_steps
        self.samples = []
        self.labels = []
        for i in tqdm(range(10)):
            sample_dir = dataset_path + '/' + str(i) + '/'
            for f in listdir(sample_dir):
                filename = join(sample_dir, f)
                if isfile(filename):
                    self.samples.append(filename)
                    self.labels.append(i)

    def __getitem__(self, index):
        data_path = self.samples[index]
        label = self.labels[index]
        tmp = np.genfromtxt(data_path, delimiter=',')

        data = np.zeros((2, 42, 42, self.n_steps))
        for c in range(2):
            for y in range(42):
                for x in range(42):
                    data[c, x, y, :] = tmp[c * 42 * 42 + y * 42 + x, :]
        data = torch.FloatTensor(data)

        return data, label

    def __len__(self):
        return len(self.samples)


def get_cifar10_dvs(data_path, network_config):
    print("loading CIFAR10 DVS")
    n_steps = network_config['n_steps']
    batch_size = network_config['batch_size']
    train_path = data_path + '/train'
    test_path = data_path + '/test'

    trainset = CIFAR10_DVS(train_path, n_steps)
    testset = CIFAR10_DVS(test_path, n_steps)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader
