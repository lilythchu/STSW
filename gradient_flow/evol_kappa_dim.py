# Adapted from: https://github.com/mint-vu/s3wd/blob/main/src/demos/evolution_runtime.ipynb 

import sys
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append('../')
from utils import vmf as vmf_utils
from methods import sswd, s3wd
from methods.stswd import stswd
from scipy.special import iv, gamma
import argparse
import os

def KL(k, d):
    cpt1 = k * iv(d/2, k)/iv(d/2-1, k)
    cpt2 = (d/2-1)*np.log(k)
    cpt3  = -np.log(2*np.pi)*d/2 - np.log(iv(d/2-1, k))
    cpt4 = np.log(np.pi)*d/2 + np.log(2) - np.log(gamma(d/2))
    return cpt1+cpt2+cpt3+cpt4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", "-l", type=str, default="stsw",
                        choices=['stsw','s3w','ri-s3w','ari-s3w','ssw','kl'],
                        help="Which loss to use")
    parser.add_argument("--ntry", type=int, default=10)
    args = parser.parse_args()
    ntry = args.ntry
    print("Method used: ", args.loss)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fig = plt.figure(figsize=(6,3))

    for d in [3,25,50,100]:
        x0 = torch.randn((500,d), device=device)
        x0 = F.normalize(x0, p=2, dim=-1)

        kappas = [1,5,10,20,30,40,50,75,100,150,200,250]
        mus = np.ones((len(kappas),d))

        if args.loss != "kl":
            L = np.zeros((len(kappas), ntry))
            for k in range(len(kappas)):
                mu = mus[k]
                mu = mu/np.linalg.norm(mu)
                x1 = vmf_utils.rand_vmf(mu,kappa=kappas[k],N=500)

                for j in range(ntry):
                    if args.loss == 's3w':
                        dist = s3wd.s3wd(x0, torch.tensor(x1, dtype=torch.float, device=device), p=2, n_projs=200, device=device) ** 0.5
                    elif args.loss == 'ri-s3w':
                        dist = s3wd.ri_s3wd(x0, torch.tensor(x1, dtype=torch.float, device=device), p=2, n_projs=200, device=device, n_rotations=100) ** 0.5
                    elif args.loss == 'ssw':
                        dist = torch.sqrt(sswd.sswd(x0, torch.tensor(x1, dtype=torch.float, device=device), p=2, num_projections=200, device=device)) ** 0.5
                    elif args.loss == 'ari-s3w':
                        dist = s3wd.ari_s3wd(x0, torch.tensor(x1, dtype=torch.float, device=device), p=2, n_projs=200, device=device, n_rotations=100, pool_size=1000) ** 0.5
                    elif args.loss == 'stsw':
                        dist = stswd(x0, torch.tensor(x1, dtype=torch.float, device=device), p=2, ntrees=200, nlines=2, device=device)

                    L[k, j] = dist

            m = np.mean(L, axis=-1)
            s = np.std(L, axis=-1)
            plt.plot(kappas, m, label=r"$d=$"+str(d)) # + r" $\mu=$["+str(mu[0])+","+str(mu[1])+","+str(mu[2])+"]")
            plt.fill_between(kappas, m-s, m+s,alpha=0.5)
        
        elif args.loss == "kl":
            L = np.zeros((len(kappas)))
            for k in range(len(kappas)):
                L[k] = KL(kappas[k], d)
            plt.plot(kappas, L, label=r"$d=$"+str(d))

    os.makedirs("figures", exist_ok=True)
    plt.xlabel(r"$\kappa$", fontsize=13)
    plt.title(rf"${args.loss.upper()}$", fontsize=13)
    plt.grid(True)
    plt.legend(fontsize=13)
    plt.savefig(f"figures/{args.loss.upper()}_vMF_Evolution.png", bbox_inches="tight")