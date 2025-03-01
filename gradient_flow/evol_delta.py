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
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", "-t", type=str, choices=['kappa', 'dim'], required=True)
    parser.add_argument("--fix_trees",  type=int, default=200)
    parser.add_argument("--fix_lines", type=int, default=10)
    parser.add_argument("--ntry", type=int, default=10)
    args = parser.parse_args()

    print(f"Type {args.type}, fix trees {args.fix_trees}, fix lines {args.fix_lines}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fig = plt.figure(figsize=(8,3))
    kappa_default = 10
    d_default = 3
    nt_default = args.fix_trees
    nl_default = args.fix_lines
    deltas = [1,2,5,10,20,25,50,75,100,150,200]

    if args.type == 'kappa':
        x0 = torch.randn((500, d_default), device=device)
        x0 = F.normalize(x0, p=2, dim=-1)

        kappas = [1, 10, 25, 100]
        mus = np.ones((len(kappas), d_default))
        L = np.zeros((len(kappas),len(deltas),args.ntry))

        for i, k in enumerate(kappas):
            mu = mus[i]
            mu = mu/np.linalg.norm(mu)
            x1 = vmf_utils.rand_vmf(mu,kappa=k,N=500)

            for l, dl in enumerate(deltas):
                for j in range(args.ntry):
                    sw = stswd(x0, torch.tensor(x1, dtype=torch.float, device=device), 
                               p=2, ntrees=nt_default, nlines=nl_default, delta=dl, device=device)
                    L[i,l,j] = sw

            m = np.mean(L[i], axis=-1)
            s = np.std(L[i], axis=-1)
            plt.plot(deltas, m, label=r"$\kappa=$"+str(k)) 
            plt.fill_between(deltas, m-s, m+s,alpha=0.5)
            
    
    elif args.type == 'dim':
        ds = [3, 10, 50, 100, 500, 1000]
        L = np.zeros((len(ds), len(deltas), args.ntry))

        for i, d in enumerate(ds):
            x0 = torch.randn((500,d), device=device)
            x0 = F.normalize(x0, p=2, dim=-1)
            
            mu = np.ones((d,))
            mu = mu/np.linalg.norm(mu)
            x1 = vmf_utils.rand_vmf(mu, kappa=kappa_default, N=500)
            
            for l, dl in enumerate(deltas):
                for j in range(args.ntry):
                    sw = stswd(x0, torch.tensor(x1, dtype=torch.float, device=device), 
                               p=2, ntrees=nt_default, nlines=nl_default, delta=dl, device=device)
                    L[i,l,j] = sw

            m = np.mean(L[i], axis=-1)
            s = np.std(L[i], axis=-1)
            plt.plot(deltas, m, label=r"$d=$"+str(d)) 
            plt.fill_between(deltas, m-s, m+s,alpha=0.5)

    ## save figs
    os.makedirs("figures", exist_ok=True)
    plt.xlabel(f"Delta", fontsize=13)
    plt.ylabel("STSW", fontsize=13)
    # plt.xscale("log")
    plt.legend(fontsize=13, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True)
    plt.savefig(f"figures/STSW_Delta_Evol_{args.type}_t{args.fix_trees}_l{args.fix_lines}.png", bbox_inches="tight")