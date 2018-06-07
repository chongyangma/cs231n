import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import timeit
import os
import argparse

from resnet import *
from vgg import *

parser = argparse.ArgumentParser(description='CIFAR-10 Image Classification')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--learning-rate', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--model_factory', type=str, default='resnet-18',
                    help="specifies which model to use")
parser.add_argument('--optim-method', type=str, default='adam',
                    help="optimization method")
args = parser.parse_args()

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

data_folder = './cs231n/datasets'

transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = dset.CIFAR10(data_folder, train=True, download=True, transform=transform_train)
loader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

test_set = dset.CIFAR10(data_folder, train=False, download=True, transform=transform_test)
loader_test = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

def train(model, loss_fn, optimizer, num_epochs = 1):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype).long())

            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            if (t + 1) % args.log_interval == 0:
                if torch.cuda.is_available():
                    print('t = %d, loss = %.6f' % (t + 1, loss.item()))
                else:
                    print('t = %d, loss = %.6f' % (t + 1, loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(dtype), requires_grad=False)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

torch.cuda.random.manual_seed(12345)


if __name__ == '__main__':
    if args.model_factory == 'resnet-18':
        model = ResNet(BasicBlock, [2, 2, 2, 2]) # ResNet-18
        checkpoint_path = 'resnet18_model.pth'
    elif args.model_factory == 'resnet-101':
        model = ResNet(Bottleneck, [3, 4, 23, 3]) # ResNet-101
        checkpoint_path = 'resnet101_model.pth'
    elif args.model_factory == 'vgg-16':
        model = VGG('VGG16') # VGG-16
        checkpoint_path = 'vgg16_model.pth'
    elif args.model_factory == 'vgg-19':
        model = VGG('VGG19') # VGG-19
        checkpoint_path = 'vgg19_model.pth'
    else:
        raise ValueError('Not a known model.')

    if torch.cuda.is_available():
        model.cuda()

    loss_fn = nn.CrossEntropyLoss().type(dtype)
    if args.optim_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    if os.path.exists(checkpoint_path):
        print('Loading checkpoint from path %s' % checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))
        check_accuracy(model, loader_test)

    train(model, loss_fn, optimizer, num_epochs=args.epochs)
    check_accuracy(model, loader_train)
    torch.save(model.state_dict(), checkpoint_path)

    best_model = model
    check_accuracy(best_model, loader_test)
