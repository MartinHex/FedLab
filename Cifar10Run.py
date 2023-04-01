from json import load
import os
import argparse
import random
from copy import deepcopy
from munch import Munch

import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate, get_best_gpu

from fedlab.models.mlp import MLP
from fedlab.models.cnn import CNN_CIFAR10
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.contrib.dataset.partitioned_cifar10 import PartitionedCIFAR10

from fedlab.utils.functional import evaluate, setup_seed
from fedlab.contrib.algorithm.fedavg import FedAvgServerHandler, FedAvgClientTrainer
from fedlab.contrib.algorithm.fedprox import FedProxServerHandler, FedProxSerialClientTrainer
from fedlab.contrib.algorithm.scaffold import ScaffoldSerialClientTrainer, ScaffoldServerHandler
from fedlab.contrib.algorithm.fednova import FedNovaSerialClientTrainer, FedNovaServerHandler
from fedlab.contrib.algorithm.feddyn import FedDynSerialClientTrainer, FedDynServerHandler

args = Munch()
args.total_client = 500
args.com_round = 20
args.sample_ratio = 0.02
args.batch_size = 20
args.epochs = 20
args.lr = 0.5

args.preprocess = False
args.seed = 0

args.alg = "fedavg"  # fedavg, fedprox, scaffold, fednova, feddyn
# optim parameter

args.mu = 0.1  # fedprox
args.alpha = 0.1  

setup_seed(args.seed)

# model = MLP(784, 10)
model = CNN_CIFAR10()

if args.alg == "fedavg":
    handler = SyncServerHandler(model=model,
        global_round=args.com_round,
        sample_ratio=args.sample_ratio)
    trainer = SGDSerialClientTrainer(model, args.total_client, cuda=True,mean_shift=False,client_shift=False)
    print(trainer.mean_shift)
    print(trainer.client_shift)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)


cifar10 = PartitionedCIFAR10(root='./datasets/cifar10/',
    path="./datasets/cifar10/fedcifar_niid",
    num_clients=args.total_client,
    partition="niid",
    dataname='cifar10',
    dir_alpha=args.alpha,
    preprocess=args.preprocess,
    transform=transforms.ToTensor())
cifar10.preprocess()
trainer.setup_dataset(cifar10)


test_data = torchvision.datasets.CIFAR10(root="./datasets/cifar10/",
    train=False,
    transform=transforms.ToTensor())

test_loader = DataLoader(test_data, batch_size=1024)

import time

for mean_sff in [True,False]:
    trainer.mean_shift=mean_sff

round = 1
accuracy = []
handler.num_clients = trainer.num_clients
while handler.if_stop is False:
    # server side
    sampled_clients = handler.sample_clients()
    broadcast = handler.downlink_package

    # client side
    trainer.local_process(broadcast, sampled_clients)
    uploads = trainer.uplink_package

    # server side
    for pack in uploads:
        handler.load(pack)

    loss, acc = evaluate(handler._model, nn.CrossEntropyLoss(), test_loader)
    accuracy.append(acc)
    print("Round {}, Test Accuracy: {:.4f}, Max Acc: {:.4f}".format(
        round, acc, max(accuracy)))
    if acc >= 0.97:
        break
    round += 1

out_path = os.path.join("./exp_logs", "Log.txt")
torch.save(accuracy,out_path)
