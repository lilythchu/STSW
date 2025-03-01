import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import time
import os
import numpy as np
from config import Config, parse_args
from model import MNISTAE, C10AE
from util import rand_unif, fibonacci_sphere
import torch.nn.functional as F
import sys
sys.path.append("../")
from methods.stswd import stswd
from methods.s3wd import ri_s3wd, ari_s3wd, s3wd
from methods.swd import swd
from methods.wd import g_wasserstein
from methods.sswd import sswd
from utils.vmf import rand_vmf, pdf_vmf
from utils.func import set_seed

def main():
    args = parse_args()
    Config.loss1 = args.loss1
    Config.loss2 = args.loss2
    Config.d = args.d
    Config.dataset = args.dataset
    Config.prior = args.prior
    Config.device = args.device
    Config.lr = args.lr
    Config.n_epochs = args.epochs
    Config.batch_size = args.batch_size
    Config.beta = args.beta
    Config.ntrees = args.ntrees
    Config.nlines = args.nlines
    Config.delta = args.delta
    Config.n_projs = args.n_projs
    Config.seed = args.seed

    set_seed(Config.seed)

    os.makedirs('results', exist_ok=True)

    if Config.dataset == 'c10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        model = C10AE(embedding_dim=Config.d)
    elif Config.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,)) 
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        model = MNISTAE(embedding_dim=Config.d)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    model = model.to(Config.device)
    criterion1 = nn.BCELoss()
    criterion2 = get_loss_func(Config.loss2, Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    start_time = time.time()
    if args.type == 'ae':
        train_ae(model, train_loader, criterion1, criterion2, optimizer, Config.device)
    elif args.type == 'swae':
        train_swae(model, train_loader, criterion1, criterion2, Config.beta, optimizer, Config.device)
    total_time = time.time() - start_time
    time_per_epoch = total_time / Config.n_epochs

    embeddings, BCE_losses = get_embs(model, test_loader, Config.device)
    avg_BCE = torch.cat(BCE_losses).mean().item()

    test_W2 = []
    test_NLL = []
    for embedding in embeddings:
        sphere_samples = get_prior(Config.prior, Config.d, embedding.size(0), Config.device)
        embedding = embedding.to(Config.device) 
        W2_dist = g_wasserstein(embedding, sphere_samples, p=2)
        test_W2.append(W2_dist)

        if Config.prior == 'vmf':
            for mu in torch.tensor(fibonacci_sphere(10), dtype=torch.float32, device=Config.device):
                nll = -torch.log(pdf_vmf(embedding, mu, kappa=10)).detach().cpu()
                test_NLL.append(nll)
        else:
            test_NLL.append(0.)

    avg_log_W2 = torch.tensor(test_W2).log().mean().item()
    avg_test_NLL = torch.tensor(test_NLL).mean().item()

    result_line = (
        f"Dataset: {Config.dataset}, "
        f"Learning Rate: {Config.lr}, "
        f"Epochs: {Config.n_epochs}, "
        f"Embedding Dim: {Config.d}, "
        f"Prior: {Config.prior}, "
        f"Loss 1: {Config.loss1}, "
        f"Loss 2: {Config.loss2}, "
        + (f"NProjs: {Config.n_projs}, " if Config.loss2 != "stsw" else
        f"Trees: {Config.ntrees}, "
        f"Lines: {Config.nlines}, "
        f"Delta: {Config.delta}, ")
        + f"Beta: {Config.beta}, "
        f"Log W2: {avg_log_W2:.4f}, "
        f"Average NLL: {avg_test_NLL:.4f}, "
        f"Average BCE: {avg_BCE:.4f}\n"
        f"Seed: {Config.seed}, "
        f"Time per Epoch: {time_per_epoch:.4f}s, "
        f"Total Time: {total_time:.4f}s\n"
    )
    with open('all_results.txt', 'a') as f:
        f.write(result_line)

def train_ae(model, train_loader, criterion1, criterion2, optimizer, device):
    for epoch in tqdm(range(Config.n_epochs), desc='Training AE'):
        for data in train_loader:
            images, _ = data
            images = images.to(device)
            optimizer.zero_grad()
            outputs, embeddings = model(images)
            loss1 = criterion1(outputs, images)
            batch_prior_samples = get_prior(Config.prior, Config.d, images.size(0), device)
            loss2 = criterion2(embeddings, batch_prior_samples)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
    save_filename = f"results/AE_{Config.dataset}_lr{Config.lr}_epoch{Config.n_epochs}_dim{Config.d}_prior{Config.prior}_loss1{Config.loss1}_loss2{Config.loss2}.pt"
    torch.save(model.state_dict(), save_filename)

def train_swae(model, train_loader, criterion1, criterion2, beta, optimizer, device):
    for epoch in tqdm(range(Config.n_epochs), desc='Training SW'):
        for data in train_loader:
            images, _ = data
            images = images.to(device)
            optimizer.zero_grad()
            outputs, embeddings = model(images)
            loss1 = criterion1(outputs, images)
            batch_prior_samples = get_prior(Config.prior, Config.d, images.size(0), device)
            loss2 = criterion2(embeddings, batch_prior_samples)
            loss = loss1 + beta * loss2
            loss.backward()
            optimizer.step()
    save_filename = f"results/SWAE_{Config.dataset}_lr{Config.lr}_epoch{Config.n_epochs}_dim{Config.d}_prior{Config.prior}_loss1{Config.loss1}_loss2{Config.loss2}_beta{Config.beta}_trees{Config.ntrees}_lines{Config.nlines}.pt"
    torch.save(model.state_dict(), save_filename)

def get_loss_func(loss_name, device):
    if loss_name == 's3w':
        return lambda X, Y: s3wd(X, Y, n_projs=Config.n_projs, p=2)
    elif loss_name.startswith('ri'):
        rotations = int(loss_name[2:])
        return lambda X, Y: ri_s3wd(X, Y, n_projs=Config.n_projs, p=2, n_rotations=rotations)
    elif loss_name.startswith('ari'):
        rotations = int(loss_name[3:])
        return lambda X, Y: ari_s3wd(X, Y, n_projs=Config.n_projs, p=2, n_rotations=rotations, pool_size=100)
    elif loss_name == 'ssw':
        return lambda X, Y: sswd(X, Y, num_projections=Config.n_projs, p=2, device=device)
    elif loss_name == 'sw':
        return lambda X, Y: swd(X, Y, n_projs=Config.n_projs, p=2, device=device)
    elif loss_name == 'mse':
        return lambda X, Y: nn.MSELoss()(X,Y)
    elif loss_name == 'stsw':
        return lambda X, Y: stswd(X, Y, ntrees=Config.ntrees, nlines=Config.nlines, p=2, 
                                  delta=Config.delta, device=device)

def get_prior(prior, dim, n_samples, device):
    if prior == 'uniform':
        return rand_unif(n_samples, dim, device)
    elif prior == 'vmf':
        assert dim == 3
        n = 10
        mus = torch.tensor(fibonacci_sphere(10), dtype=torch.float32)
        kappa = 10

        ps = np.ones(n)/n
        Z = np.random.multinomial(n_samples,ps)
        X = []
        for k in range(len(Z)):
            if Z[k]>0:
                vmf = rand_vmf(mus[k], kappa=kappa, N=int(Z[k]))
                X += list(vmf)
        z = torch.tensor(X, device=device, dtype=torch.float)
        return z

def get_embs(model, data_loader, device):
    model.eval()
    embeddings = []
    BCE_losses = []
    with torch.no_grad():
        for data in data_loader:
            images, _ = data
            images = images.to(device)
            outputs, embedding = model(images)
            images = images.clamp(0, 1)
            outputs = outputs.clamp(0, 1)
            BCE_loss = F.binary_cross_entropy(outputs, images, reduction='none')
            BCE_loss = BCE_loss.mean(dim=[1, 2, 3]).detach().cpu()
            BCE_losses.append(BCE_loss)
            embeddings.append(embedding.detach().cpu())
    return embeddings, BCE_losses

if __name__ == '__main__':
    main()
