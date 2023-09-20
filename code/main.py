import utils
SEED = 2021
utils.set_seed(SEED)

from dataloader import GetData, ALL_TRAIN
from model import IGCNSDA
import torch
import numpy as np
import time
from tqdm import tqdm

dataset = GetData(path="../data/MNDR")
device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
latent_dim = 256
n_layers = 3
lr = 0.001

model = IGCNSDA(dataset, latent_dim=latent_dim, n_layers=n_layers, groups=3, dropout_bool=False, l2_w=0.0002, single=True).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

EPOCHS = 100
BATCH_SIZE = 1024

def run_train(dataset, model, optimiser):
    model.train()
    num_batches = dataset.trainSize // BATCH_SIZE + 1
    mean_batch_loss = 0

    S = utils.UniformSample(dataset)
    users = torch.Tensor(S[:, 0]).long().to(device)
    posItems = torch.Tensor(S[:, 1]).long().to(device)
    negItems = torch.Tensor(S[:, 2]).long().to(device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)

    for i,(b_users, b_pos, b_neg) in enumerate(utils.minibatch((users,posItems,negItems), BATCH_SIZE)):
        loss = model.bpr_loss(b_users, b_pos, b_neg)
        mean_batch_loss += loss.cpu().item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return f"Final train loss: {mean_batch_loss/num_batches:.7f}"

def main():
    for epoch in range(1,EPOCHS+1):
        train_info = run_train(dataset, model, optimiser)
        print(train_info)


if __name__ == "__main__":
    main()
    