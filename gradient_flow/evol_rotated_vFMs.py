# Adapted from: https://github.com/mint-vu/s3wd/blob/main/src/demos/evolution_runtime.ipynb 

import sys
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append('../')
from utils import vmf as vmf_utils
from methods import sswd, s3wd, swd
from methods.stswd import stswd
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", "-l", type=str, default="stsw",
                        choices=['stsw','s3w','ri-s3w','ari-s3w','ssw'],
                        help="Which loss to use")
    parser.add_argument("--ntry", type=int, default=10)
    args = parser.parse_args()
    ntry = args.ntry
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kappa = 10    
    ds = [3, 25, 50]
    thetas = [0,np.pi/6,np.pi/3,np.pi/2,2*np.pi/3,5*np.pi/6,np.pi,
                -5*np.pi/6,-2*np.pi/3,-np.pi/2,-np.pi/3,-np.pi/6, 0]
    thetas = np.array(thetas)%(2*np.pi)
    thetas[-1] = 2*np.pi

    L = np.zeros((len(ds), len(thetas), ntry))

    fig = plt.figure(figsize=(6,3))

    for i, d in enumerate(ds):
        mu_target = np.zeros((d,))
        mu_target[0] = 1
        
        v = np.zeros((d,))
        v[1] = 1

        mus = []
        for theta in thetas:
            mus.append(np.zeros((d,)))
            mus[-1][0] = np.cos(theta)
            mus[-1][1] = np.sin(theta)  
            
        for k in range(len(mus)):
            mu = np.array(mus[k])
            
            for j in range(ntry):
                x0 = vmf_utils.rand_vmf(mu_target, kappa=10, N=500)
                x1 = vmf_utils.rand_vmf(mu, kappa=10, N=500)
                if args.loss == 's3w':
                    sw = s3wd.s3wd(torch.tensor(x0, dtype=torch.float, device=device), 
                            torch.tensor(x1, dtype=torch.float, device=device), 
                            n_projs=200, device=device, p=2) ** 0.5
                elif args.loss == 'ri-s3w':
                    sw = s3wd.ri_s3wd(torch.tensor(x0, dtype=torch.float, device=device), 
                            torch.tensor(x1, dtype=torch.float, device=device), 
                            n_projs=200, n_rotations=100, device=device, p=2) ** 0.5
                elif args.loss == 'ari-s3w':
                    sw = s3wd.ari_s3wd(torch.tensor(x0, dtype=torch.float, device=device), 
                            torch.tensor(x1, dtype=torch.float, device=device), 
                            n_projs=200, n_rotations=100, device=device, p=2, pool_size=1000) ** 0.5
                elif args.loss == 'ssw':
                    sw = sswd.sswd(torch.tensor(x0, dtype=torch.float, device=device), 
                            torch.tensor(x1, dtype=torch.float, device=device), 
                            num_projections=200, device=device, p=2) ** 0.5
                elif args.loss == 'sw':
                    sw = swd.swd(torch.tensor(x0, dtype=torch.float, device=device), 
                          torch.tensor(x1, dtype=torch.float, device=device), 
                          n_projs=200, device=device, p=2) ** 0.5
                elif args.loss == 'stsw':
                    sw = stswd(torch.tensor(x0, dtype=torch.float, device=device), 
                             torch.tensor(x1, dtype=torch.float, device=device), 
                             ntrees=200, nlines=2, device=device, p=2)

                L[i, k, j] = sw

        m = np.mean(L[i], axis=-1)
        s = np.std(L[i], axis=-1)
        plt.plot(thetas, m, label=r"$d=$"+str(d))
        plt.fill_between(thetas, m-s, m+s,alpha=0.5)
    
    os.makedirs("figures", exist_ok=True)
    plt.xlabel(r"$\theta$") 
    labels = ["0", r"$\frac{\pi}{6}$", r"$\frac{\pi}{3}$", r"$\frac{\pi}{2}$", r"$\frac{2\pi}{3}$", r"$\frac{5\pi}{6}$",
                r"$\pi$", r"$\frac{7\pi}{6}$", r"$\frac{4\pi}{3}$", r"$\frac{3\pi}{2}$", r"$\frac{5\pi}{3}$",
                r"$\frac{11\pi}{6}$", r"$2\pi$"]
    plt.xticks(thetas, labels, fontsize=10)
    plt.title(rf"${args.loss.upper()}$", fontsize=13)
    plt.grid(True)
    plt.legend(fontsize=13, loc="upper right")
    plt.savefig(f"figures/{args.loss.upper()}_vMF_Rotation.png", bbox_inches="tight")