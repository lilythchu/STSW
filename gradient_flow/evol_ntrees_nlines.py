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
    parser.add_argument("--type", "-t", type=str, choices=['nt-kappa', 'nt-dim', 'nl-kappa', 'nl-dim'], required=True)
    parser.add_argument("--fix_trees",  type=int, default=10)
    parser.add_argument("--fix_lines", type=int, default=10)
    parser.add_argument("--ntry", type=int, default=20)
    args = parser.parse_args()

    print(f"Type {args.type}, fix trees {args.fix_trees}, fix lines {args.fix_lines}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fig = plt.figure(figsize=(8,3))
    kappa_default = 10
    d_default = 3
    title = None

    if args.type == 'nt-kappa':
        nl_default = args.fix_lines
        x0 = torch.randn((500, d_default), device=device)
        x0 = F.normalize(x0, p=2, dim=-1)

        kappas = [1, 10, 25, 100]
        mus = np.ones((len(kappas), d_default))
        num_trees = [1,10,50,100,200,400,500,750,900,1000]
        L = np.zeros((len(kappas),len(num_trees),args.ntry))

        for i, k in enumerate(kappas):
            mu = mus[i]
            mu = mu/np.linalg.norm(mu)
            x1 = vmf_utils.rand_vmf(mu,kappa=k,N=500)

            for l, nt in enumerate(num_trees):
                for j in range(args.ntry):
                    sw = stswd(x0, torch.tensor(x1, dtype=torch.float, device=device), 
                               p=2, ntrees=nt, nlines=nl_default, device=device)
                    L[i,l,j] = sw

            m = np.mean(L[i], axis=-1)
            s = np.std(L[i], axis=-1)
            
            plt.plot(num_trees, m, label=r"$\kappa=$"+str(k)) 
            plt.fill_between(num_trees, m-s, m+s,alpha=0.3)
            title = f"Fixed Num Lines = {nl_default}"
    
    elif args.type == 'nt-dim':
        nl_default = args.fix_lines
        ds = [3, 10, 50, 100, 500, 1000]
        num_trees = [1,10,50,100,200,400,500,600,700,750]

        L = np.zeros((len(ds), len(num_trees), args.ntry))

        for i, d in enumerate(ds):
            x0 = torch.randn((500,d), device=device)
            x0 = F.normalize(x0, p=2, dim=-1)
            
            mu = np.ones((d,))
            mu = mu/np.linalg.norm(mu)
            x1 = vmf_utils.rand_vmf(mu, kappa=kappa_default, N=500)
            
            for l, nt in enumerate(num_trees):
                for j in range(args.ntry):
                    sw = stswd(x0, torch.tensor(x1, dtype=torch.float, device=device), 
                               p=2, ntrees=nt, nlines=nl_default, device=device)
                    L[i,l,j] = sw

            m = np.mean(L[i], axis=-1)
            s = np.std(L[i], axis=-1)

            plt.plot(num_trees, m, label=r"$d=$"+str(d)) 
            plt.fill_between(num_trees, m-s, m+s,alpha=0.5)
            title = f"Fixed Num Lines = {nl_default}"

    elif args.type == 'nl-kappa':
        nt_default = args.fix_trees
        x0 = torch.randn((500, d_default), device=device)
        x0 = F.normalize(x0, p=2, dim=-1)

        kappas = [1, 10, 25, 100]
        mus = np.ones((len(kappas), d_default))
        num_lines = [1,10,50,100,200,400,500,600,700,750]

        L = np.zeros((len(kappas),len(num_lines),args.ntry))
        for i, k in enumerate(kappas):
            mu = mus[i]
            mu = mu/np.linalg.norm(mu)
            x1 = vmf_utils.rand_vmf(mu,kappa=k,N=500)

            for l, nl in enumerate(num_lines):
                for j in range(args.ntry):
                    sw = stswd(x0, torch.tensor(x1, dtype=torch.float, device=device), 
                               p=2, ntrees=nt_default, nlines=nl, device=device)
                    L[i,l,j] = sw

            m = np.mean(L[i], axis=-1)
            s = np.std(L[i], axis=-1)
            
            plt.plot(num_lines, m, label=r"$\kappa=$"+str(k)) # + r" $\mu=$["+str(mu[0])+","+str(mu[1])+","+str(mu[2])+"]")
            plt.fill_between(num_lines, m-s, m+s,alpha=0.3)
            title = f"Fixed Num Trees = {nt_default}"

    elif args.type == 'nl-dim':
        nt_default = args.fix_trees
        ds = [3, 10, 50, 100, 500, 1000]
        num_lines = [1,10,50,100,200,400,500,600,700,750]

        L = np.zeros((len(ds), len(num_lines), args.ntry))

        for i, d in enumerate(ds):
            x0 = torch.randn((500,d), device=device)
            x0 = F.normalize(x0, p=2, dim=-1)
            
            mu = np.ones((d,))
            mu = mu/np.linalg.norm(mu)
            x1 = vmf_utils.rand_vmf(mu, kappa=kappa_default, N=500)
            
            for l, nl in enumerate(num_lines):
                for j in range(args.ntry):
                    sw = stswd(x0, torch.tensor(x1, dtype=torch.float, device=device), 
                               p=2, ntrees=nt_default, nlines=nl, device=device)
                    L[i,l,j] = sw

            m = np.mean(L[i], axis=-1)
            s = np.std(L[i], axis=-1)

            plt.plot(num_lines, m, label=r"$d=$"+str(d)) 
            plt.fill_between(num_lines, m-s, m+s,alpha=0.5)
            title = f"Fixed Num Trees = {nt_default}"

    ## save figs
    metric = "Trees" if args.type.startswith("nt") else "Lines"
    os.makedirs("figures", exist_ok=True)
    plt.xlabel(f"Number of {metric}", fontsize=13)
    # plt.ylabel(r"$STSW$", fontsize=13, rotation=0, labelpad=20)
    plt.ylabel("STSW", fontsize=13)
    plt.xscale("log")
    # plt.legend(fontsize=13, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", ncol=2)
    plt.legend(fontsize=13, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True)
    plt.title(title, fontsize=13)
    plt.savefig(f"figures/STSW_{metric}_Evol_{args.type}_t{args.fix_trees}_l{args.fix_lines}.png", bbox_inches="tight")