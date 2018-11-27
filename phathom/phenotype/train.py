import os
from itertools import cycle
import torch
import numpy as np
import tifffile
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.autograd import Variable
from torchvision.datasets import DatasetFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from .models import AuxiliaryDeepGenerativeModel
from .inference import SVI, DeterministicWarmup, ImportanceWeightedSampler
from .utils import onehot

working_dir = '/home/jswaney/pv_gabi'

cuda = torch.cuda.is_available()
np.random.seed(333)

y_dim = 2
z_dim = 100
a_dim = 100
h_dim = [500, 500]

model = AuxiliaryDeepGenerativeModel([1024, y_dim, z_dim, a_dim, h_dim])


def tiff_loader(path):
    img = tifffile.imread(path).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    img = img[..., np.newaxis]
    return img


def get_dataloaders(working_dir, subset_frac, valid_split, batch_size, nb_workers):
    flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()

    unlabeled_dataset = DatasetFolder(root=os.path.join(working_dir, 'unlabeled'),
                                      loader=tiff_loader,
                                      extensions=['.tif'],
                                      transform=flatten_bernoulli)
    labeled_dataset = DatasetFolder(root=os.path.join(working_dir, 'labeled'),
                                    loader=tiff_loader,
                                    extensions=['.tif'],
                                    transform=flatten_bernoulli,
                                    target_transform=onehot(y_dim))

    nb_unlabeled = len(unlabeled_dataset)
    nb_subset = int(subset_frac * nb_unlabeled)
    idx = np.arange(nb_unlabeled)
    np.random.shuffle(idx)
    idx_subset = idx[:nb_subset]

    nb_labeled = len(labeled_dataset)
    nb_valid = int(valid_split * nb_labeled)

    idx = np.arange(nb_labeled)
    np.random.shuffle(idx)
    idx_valid = idx[:nb_valid]
    idx_train = idx[nb_valid:]

    print("Unlabeled subset: {}".format(nb_subset))
    print("Training subset: {}".format(nb_labeled - nb_valid))
    print("Validation subset: {}".format(nb_valid))

    unlabeled = DataLoader(unlabeled_dataset,
                           batch_size=batch_size,
                           sampler=SubsetRandomSampler(idx_subset),
                           num_workers=nb_workers,
                           pin_memory=cuda)

    labeled = DataLoader(labeled_dataset,
                         batch_size=batch_size,
                         sampler=SubsetRandomSampler(idx_train),
                         num_workers=nb_workers,
                         pin_memory=cuda)

    validation = DataLoader(labeled_dataset,
                            batch_size=batch_size,
                            sampler=SubsetRandomSampler(idx_valid),
                            num_workers=nb_workers,
                            pin_memory=cuda)

    return unlabeled, labeled, validation


unlabeled, labeled, validation = get_dataloaders(working_dir, subset_frac=0.1, valid_split=0.5, batch_size=64, nb_workers=0)


beta = 0.1
alpha = beta * (len(unlabeled) + len(labeled)) / len(labeled)


def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
beta = DeterministicWarmup(n=200)
sampler = ImportanceWeightedSampler(mc=10)

if cuda: model = model.cuda()
elbo = SVI(model, likelihood=binary_cross_entropy, beta=beta, sampler=sampler)


alpha = 0
for epoch in range(100):
    model.train()
    total_loss, accuracy = (0, 0)
    for (x, y), (u, _) in zip(cycle(labeled), unlabeled):
        u = Variable(u)

        if cuda:
            # They need to be on the same device and be synchronized.
            x, y = x.cuda(device=0), y.cuda(device=0)
            u = u.cuda(device=0)

        L = -elbo(x, y)
        U = -elbo(u)
        # Add auxiliary classification loss q(y|x)
        logits = model.classify(x)
        # Regular cross entropy
        classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

        J_alpha = L - alpha * classication_loss + U
        J_alpha.backward()
        optimizer.step()

        optimizer.zero_grad()

        total_loss += J_alpha.item()
        accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())

    if epoch % 1 == 0:
        model.eval()
        m = len(unlabeled)
        print("Epoch: {}".format(epoch))
        print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))

        total_loss, accuracy = (0, 0)
        for x, y in validation:
            x, y = Variable(x), Variable(y)

            if cuda:
                x, y = x.cuda(device=0), y.cuda(device=0)

            L = -elbo(x, y)
            U = -elbo(x)

            logits = model.classify(x)
            classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

            J_alpha = L + alpha * classication_loss + U

            total_loss += J_alpha.item()

            _, pred_idx = torch.max(logits, 1)
            _, lab_idx = torch.max(y, 1)
            accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())

        m = len(validation)
        print("[Validation]\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))