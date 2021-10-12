import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
from torch_cka import CKA
from timm.models import vit_base_patch32_224, resnetv2_50x1_bitm

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

model1 = resnetv2_50x1_bitm(pretrained=True)
model2 = vit_base_patch32_224(pretrained=True)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 256

dataset = CIFAR10(root='../data/',
                  train=False,
                  download=True,
                  transform=transform)

dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g,)

cka = CKA(model1, model2,
          device='cuda')

cka.compare(dataloader)

cka.plot_results(save_path="../assets/resnet-vit_compare.png")