# Adapted from: https://github.com/mint-vu/s3wd/blob/main/src/demos/gf_runtime.ipynb

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
from itertools import cycle
from scipy.stats import gaussian_kde
from tqdm.auto import trange
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
import utils.vmf as vmf_utils
import utils.plot as plot_utils
from methods import s3wd, sswd, wd, stswd

def vmf_pdf(x, mu, kappa):
    kappa = torch.tensor(kappa, dtype=torch.float32, device=x.device)
    mu = mu.to(x.device)
    C_d = kappa / (2 * np.pi * (torch.exp(kappa) - torch.exp(-kappa)))
    return C_d * torch.exp(kappa * torch.matmul(x, mu.T))

def plot_result(X, out_path):
    k = gaussian_kde(X.T)
    fig, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw={'projection': "mollweide"})
    plot_utils.projection_mollweide(lambda x: k.pdf(x.T), ax)
    plt.savefig(out_path)
    plt.close(fig)

def run_exp(dataiter, d_func, d_args, n_steps=500, lr=1e-2, kappa=50, device='cpu', eval=False):
    X0 = torch.randn((args.batch_size, 3), device=device)
    X0 = F.normalize(X0, p=2, dim=-1)
    X0.requires_grad_(True)

    optimizer = torch.optim.Adam([X0], lr=lr)
    loss_w = []

    tic = time.time()
    pbar = trange(n_steps)
    for _ in pbar:
        optimizer.zero_grad()
        Xt = next(dataiter).to(device)
        sw = d_func(Xt, X0, **d_args)
        sw.backward()
        optimizer.step()
        X0.data /= torch.norm(X0.data, dim=1, keepdim=True)
        pbar.set_description(f"Loss: {sw.item():.4f}")
        
        if eval:
            with torch.no_grad():
                w = wd.g_wasserstein(X0, Xt, p=2, device=device)    
                loss_w.append(w.item())
            
    pbar.close()
    t = time.time() - tic
    X0_np = X0.detach().cpu().numpy()
    Xt_np = Xt.detach().cpu().numpy()

    w = wd.g_wasserstein(torch.tensor(X0_np, device=device), torch.tensor(Xt_np, device=device), p=2, device=device)    
    log_probs = torch.stack([vmf_pdf(X0, mu, kappa) for mu in mus])
    log_sum_probs = torch.logsumexp(log_probs, dim=0) - torch.log(torch.tensor(len(mus), device=device))
    nll = -torch.sum(log_sum_probs).item()

    return X0_np, t, nll, w.item(), loss_w

def summary_result(args, results):
    runtimes = [r[1] for r in results]
    nll=[r[2] for r in results]
    w = [r[3] for r in results]
    log_wd = np.log10(w)
    if args.eval:
        evol_loss = [r[4] for r in results]
        np.save(f'results/{get_run_name(args)}.npy', np.mean(evol_loss, axis=0))

    to_print = f"\tRuntime: Mean = {np.mean(runtimes)}, Std = {np.std(runtimes)}, Min = {np.min(runtimes)}, Max = {np.max(runtimes)}\n\
    NLL: Mean = {np.mean(nll)}, Std = {np.std(nll)}, Min = {np.min(nll)}, Max = {np.max(nll)}\n\
    Log Wasserstein: Mean = {np.mean(log_wd)}, Std = {np.std(log_wd)}, Min = {np.min(log_wd)}, Max = {np.max(log_wd)}\n"

    print(to_print)
    with open(f"all_results.txt", "a") as f:
        f.write(get_run_name(args) + f": eval {args.eval}\n")
        f.write(to_print)

    best = np.argmin(w)
    X0_best = results[best][0]
    plot_result(X0_best, f"figures/{get_run_name(args)}.png")

def get_run_name(args):
    if args.d_func == "stsw":
        return f"{args.d_func}-nt_{args.ntrees}-nl_{args.nlines}-lr_{args.lr}-try{args.ntry}"
    return f"{args.d_func}-np_{args.n_projs}-lr_{args.lr}-try{args.ntry}"
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('gradient flow parameters')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--d_func', '-d', type=str, default="stsw")
    parser.add_argument('--ntry', type=int, default=10)
    parser.add_argument('--ntrees', '-nt', type=int, default=200)
    parser.add_argument('--nlines', '-nl', type=int, default=5)
    parser.add_argument('--n_projs', '-np', type=int, default=1000)
    parser.add_argument('--epochs', '-ep', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=2400)
    parser.add_argument('--eval', action='store_true',default=False)
    
    args = parser.parse_args()
    device = args.device
    os.makedirs('figures', exist_ok=True)

    phi = (1 + np.sqrt(5)) / 2
    vs = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1]
    ])
    mus = F.normalize(torch.tensor(vs, dtype=torch.float), p=2, dim=-1)
    X = []
    kappa = 50 
    N = 200   
    for mu in mus:
        vmf = vmf_utils.rand_vmf(mu, kappa=kappa, N=N)
        X += list(vmf)
    X = torch.tensor(X, dtype=torch.float)
    Xt = X.clone().detach()
    trainloader = DataLoader(Xt, batch_size=args.batch_size, shuffle=True)
    dataiter = iter(cycle(trainloader))

    if args.d_func == "stsw":
        d_func = stswd.stswd
        d_args = {'p': 2, 'ntrees': args.ntrees, 'nlines': args.nlines, 'device': device}
    elif args.d_func == "ari_s3w":
        d_func = s3wd.ari_s3wd
        d_args = {'p': 2, 'n_projs': args.n_projs, 'device': device, 'h': None, 'n_rotations': 30, 'pool_size': 1000}
    elif args.d_func == "s3w":
        d_func = s3wd.s3wd
        d_args = {'p': 2, 'n_projs': args.n_projs, 'device': device, 'h': None}
    elif args.d_func == "ri_s3w_1":
        d_func = s3wd.ri_s3wd
        d_args = {'p': 2, 'n_projs': args.n_projs, 'device': device, 'h': None, 'n_rotations': 1}
    elif args.d_func == "ri_s3w_5":
        d_func = s3wd.ri_s3wd
        d_args = {'p': 2, 'n_projs': args.n_projs, 'device': device, 'h': None, 'n_rotations': 5}
    elif args.d_func == "ssw":
        d_func = sswd.sswd
        d_args = {'p': 2, 'num_projections': args.n_projs, 'device': device}
    else:
        raise Exception(f"Loss function {args.d_func} is not supported")
    
    results = [run_exp(dataiter, d_func, d_args, lr=args.lr, n_steps=args.epochs, device=device, eval=args.eval) for _ in range(args.ntry)]
    summary_result(args, results)