# Adapted from: https://github.com/mint-vu/s3wd/blob/main/src/demos/evolution_runtime.ipynb 

import sys
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append('../')
from utils import vmf as vmf_utils
from methods.stswd import stswd
import argparse
import time
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", "-t", type=str, choices=['ntrees', 'nlines', 'nsamples'])
    parser.add_argument("--ntry", type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fig = plt.figure(figsize=(8,3))
    kappa = 10
    ds = [3, 10, 100, 300, 500, 1000]
    nt_default = 200
    nl_default = 10
    
    if args.type == "nsamples":
        samples = [500,1000,3000,5000,7000,8000,10000]
        L = np.zeros((len(ds), len(samples), args.ntry))

        for i, d in enumerate(ds):
            mu = np.ones((d,))
            mu = mu/np.linalg.norm(mu)

            for k, n_samples in enumerate(samples):   
                x0 = torch.randn((n_samples, d), device=device)
                x0 = F.normalize(x0, p=2, dim=-1)
                x1 = vmf_utils.rand_vmf(mu, kappa=kappa, N=n_samples)
                for j in range(args.ntry):
                    t0 = time.time()
                    sw = stswd(x0, torch.tensor(x1, dtype=torch.float, device=device), p=2, 
                                     ntrees=nt_default, nlines=nl_default, device=device)
                    L[i,k,j] = time.time()-t0

            m = np.mean(L[i][:, 1:], axis=-1)
            s = np.std(L[i][:, 1:], axis=-1)

            plt.plot(samples, m, label=r"$d=$"+str(d))
            plt.fill_between(samples, m-s, m+s,alpha=0.5)
            plt.xlabel("Number of samples", fontsize=13)
            plt.ylabel("Seconds", fontsize=13)

    elif args.type == 'ntrees':
        num_trees = [200,400,500,750,900,1000,1250,1500,1750,2000]
        L = np.zeros((len(ds), len(num_trees), args.ntry))

        for i, d in enumerate(ds):
            x0 = torch.randn((500, d), device=device)
            x0 = F.normalize(x0, p=2, dim=-1)
            
            mu = np.ones(( d,))
            mu = mu/np.linalg.norm(mu)
            x1 = vmf_utils.rand_vmf(mu, kappa=kappa, N=500)
            
            for l, nt in enumerate(num_trees):
                for j in range(args.ntry):
                    t0 = time.time()
                    try:
                        sw = stswd(x0, torch.tensor(x1, dtype=torch.float, device=device), p=2,
                                         ntrees=nt, nlines=nl_default, device=device)
                        L[i,l,j] = time.time()-t0
                    except:
                        L[i, l, j] = np.inf

            m = np.mean(L[i][:, 1:], axis=-1)
            s = np.std(L[i][:, 1:], axis=-1)

            plt.plot(num_trees, m, label=r"$d=$"+str(d))
            plt.fill_between(num_trees, m-s, m+s,alpha=0.5)
            plt.xlabel("Number of trees", fontsize=13)
            plt.ylabel("Seconds", fontsize=13)
    
    elif args.type == 'nlines':
        num_lines = [5,10,25,50,100,150,200,300,500,750,1000]
        L = np.zeros((len(ds), len(num_lines), args.ntry))

        for i, d in enumerate(ds):
            x0 = torch.randn((500, d), device=device)
            x0 = F.normalize(x0, p=2, dim=-1)
            
            mu = np.ones(( d,))
            mu = mu/np.linalg.norm(mu)
            x1 = vmf_utils.rand_vmf(mu, kappa=kappa, N=500)
            
            for l, nl in enumerate(num_lines):
                for j in range(args.ntry):
                    t0 = time.time()
                    try:
                        sw = stswd(x0, torch.tensor(x1, dtype=torch.float, device=device), p=2,
                                         ntrees=nt_default, nlines=nl, device=device)
                        L[i,l,j] = time.time()-t0
                    except:
                        L[i, l, j] = np.inf

            m = np.mean(L[i][:, 1:], axis=-1)
            s = np.std(L[i][:, 1:], axis=-1)

            plt.plot(num_lines, m, label=r"$d=$"+str(d))
            plt.fill_between(num_lines, m-s, m+s,alpha=0.5)
            plt.xlabel("Number of lines", fontsize=13)
            plt.ylabel("Seconds", fontsize=13)

    
    ## save figs
    os.makedirs("figures", exist_ok=True)
    plt.legend(fontsize=13, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"figures/STSW_time_{args.type}.png", bbox_inches="tight")
