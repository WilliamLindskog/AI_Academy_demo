from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union
import os
import random
import warnings

import flwr as fl
import torch
import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore')

DEVICE = torch.device("cpu")
NUM_CLIENTS = 2
NUM_ROUNDS = 5

def load_data() -> Union[DataLoader, DataLoader]:
    """ Load data CIFAR10 from torchvision.datasets """
    # Set transform
    transform = transforms.Compose(
        [
            # cifar10 optimal transform
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.49139968, 0.48215841, 0.44653091],
                std=[0.24703223, 0.24348513, 0.26158784],
            ),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )

    # Set to data loader
    # load 10% randomly sampled training and test dataset
    train_dataset = torch.utils.data.Subset(train_dataset, random.sample(range(len(train_dataset)), int(0.1 * len(train_dataset))))
    test_dataset = torch.utils.data.Subset(test_dataset, random.sample(range(len(test_dataset)), int(0.1 * len(test_dataset))))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader

# Get simple MLP model
def get_model() -> torch.nn.Module:
    """ Get ResNet18 model """
    # simple mlp model for cifar10
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = torch.nn.Flatten()
            self.linear_relu_stack = torch.nn.Sequential(
                torch.nn.Linear(32 * 32 * 3, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 10),
                torch.nn.ReLU(),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
        
    return MLP()


def train(net, trainloader, epochs):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    net.train()
    for _ in range(epochs):
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader):
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = 0
    net.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            loss += loss_fn(outputs, targets).item()
            predictions = torch.argmax(outputs, dim=-1)
            total += targets.size(0)
            correct += (predictions == targets).sum().item()
    accuracy = correct / total
    loss /= len(testloader.dataset)
    return loss, accuracy

class CIFAR10Client(fl.client.NumPyClient):
    def __init__(self,):
        self.net = get_model().to(DEVICE)
        self.trainloader, self.testloader = load_data()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        print("-------------- State dict ------------------: ", state_dict)
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("Training Started...")
        train(self.net, self.trainloader, epochs=1)
        print("Training Finished.")
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy), "loss": float(loss)}

def client_fn(cid):
    return CIFAR10Client()

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    print("accuracy: ", sum(accuracies) / sum(examples))
    return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=weighted_average,
)

hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 1},
    ray_init_args={"log_to_driver": False, "num_cpus": 1, "num_gpus": 1}
)

metric_type = "distributed"
metric_dict = (
    hist.metrics_centralized
    if metric_type == "centralized"
    else hist.metrics_distributed
)
_, values = zip(*metric_dict["accuracy"])

# let's extract decentralized loss (main metric reported in FedProx paper)
rounds_loss, values_loss = zip(*hist.losses_distributed)

_, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
axs[1].plot(np.asarray(rounds_loss), np.asarray(values))

axs[0].set_ylabel("Loss")
axs[1].set_ylabel("Accuracy")

# plt.title(f"{metric_type.capitalize()} Validation - MNIST")
plt.xlabel("Rounds")
# plt.legend(loc="lower right")

plt.show()