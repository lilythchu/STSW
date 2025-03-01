import sys
import torch
import numpy as np
import torch.nn.functional as F
import ot
import time
import matplotlib.pyplot as plt

sys.path.append('../')
from utils import vmf as vmf_utils
from utils.s3w import RotationPool
from methods import sswd as ssw, s3wd as s3w
from methods import swd as sw2
from methods.stswd import  stswd

import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ntry = 15
kappa = 10
pool_size = 100
n_rot = 10
ds = [3]

samples = [100, 300, 500, 1000, 5000, int(1e4), int(1e5)]
projs = [200]

L_stswd = np.zeros((len(ds), len(projs), len(samples), ntry))
L_aris3wd = np.zeros((len(ds), len(projs), len(samples), ntry))
L_ris3wd = np.zeros((len(ds), len(projs), len(samples), ntry))
L_s3wd = np.zeros((len(ds), len(projs), len(samples), ntry))
L_ssw2 = np.zeros((len(ds), len(projs), len(samples), ntry))
L_unif = np.zeros((len(ds), len(projs), len(samples), ntry))
L_ssw1 = np.zeros((len(ds), len(projs), len(samples), ntry))
L_sw2 = np.zeros((len(ds), len(projs), len(samples), ntry))

L_w = np.zeros((len(ds), len(samples), ntry))
L_s = np.zeros((len(ds), len(samples), ntry))

for i, dim in enumerate(ds):
    print(dim, flush=True)
    
    mu = np.ones((dim,))
    mu = mu/np.linalg.norm(mu)
    
    for k, n_samples in enumerate(samples):
        print(n_samples, flush=True)
        x0 = torch.randn((n_samples, dim), device=device)
        x0 = F.normalize(x0, p=2, dim=-1)
    
        x1 = vmf_utils.rand_vmf(mu, kappa=kappa, N=n_samples)

        for j in range(ntry):
            for l, n_projs in enumerate(projs):

                try:
                    t0 = time.time()
                    d = stswd(x0, torch.tensor(x1, dtype=torch.float, device=device), p=2, ntrees=200, nlines=10, device=device)
                    L_stswd[i,l,k,j] = time.time()-t0
                except Exception as e:
                    print(e)
                    L_stswd[i,l,k,j] = np.inf

                gc.collect()
                torch.cuda.empty_cache()

                try:
                    t0 = time.time()
                    d = s3w.s3wd(x0, torch.tensor(x1, dtype=torch.float, device=device), p=2, n_projs=n_projs, device=device)
                    L_s3wd[i,l,k,j] = time.time()-t0
                except Exception as e:
                    print(e)
                    L_s3wd[i,l,k,j] = np.inf

                gc.collect()
                torch.cuda.empty_cache()
                

                try:
                    t0 = time.time()
                    d = s3w.ri_s3wd(x0, torch.tensor(x1, dtype=torch.float, device=device), p=2, n_projs=n_projs, device=device, n_rotations=n_rot)
                    L_ris3wd[i,l,k,j] = time.time()-t0
                except Exception as e:
                    print(e)
                    L_ris3wd[i,l,k,j] = np.inf

                gc.collect()
                torch.cuda.empty_cache()

                try:
                    t0 = time.time()
                    d = s3w.ari_s3wd(x0, torch.tensor(x1, dtype=torch.float, device=device), p=2, n_projs=n_projs, device=device, n_rotations=n_rot, pool_size=pool_size)
                    L_aris3wd[i,l,k,j] = time.time()-t0
                except Exception as e:
                    print(e)
                    L_aris3wd[i,l,k,j] = np.inf

                gc.collect()
                torch.cuda.empty_cache()

                try:
                    t0 = time.time()
                    d = sw2.swd(x0, torch.tensor(x1, dtype=torch.float, device=device), p=2, n_projs=n_projs, device=device)
                    L_sw2[i,l,k,j] = time.time()-t0
                except Exception as e:
                    print(e)
                    L_sw2[i,l,k,j] = np.inf

                gc.collect()
                torch.cuda.empty_cache()

                try:
                    t0 = time.time()
                    d = ssw.sswd(x0, torch.tensor(x1, dtype=torch.float, device=device), n_projs, device, p=2)
                    L_ssw2[i,l,k,j] = time.time()-t0
                except:
                    L_ssw2[i,l,k,j] = np.inf
                
                gc.collect()
                torch.cuda.empty_cache()
                    
                try:
                    t0 = time.time()
                    d = ssw.sswd(x0, torch.tensor(x1, dtype=torch.float, device=device), n_projs, device, p=1)
                    L_ssw1[i,l,k,j] = time.time()-t0
                except:
                    L_ssw1[i,l,k,j] = np.inf
                
                gc.collect()
                torch.cuda.empty_cache()

                try:
                    t0 = time.time()
                    d = ssw.sswd_unif(torch.tensor(x1, dtype=torch.float, device=device), n_projs, device)
                    L_unif[i,l,k,j] = time.time()-t0
                except:
                    L_unif[i,l,k,j] = np.inf

                gc.collect()
                torch.cuda.empty_cache()

            if n_samples > 2e4:
                L_s[i,k,j] = np.inf
                L_w[i,k,j] = np.inf
                continue

            try:
                t2 = time.time()
                ip = x0@torch.tensor(x1, dtype=torch.float, device=device).T
                M = torch.arccos(torch.clamp(ip, min=-1+1e-5, max=1-1e-5))
                a = torch.ones(x0.shape[0], device=device) / x0.shape[0]
                b = torch.ones(x1.shape[0], device=device) / x1.shape[0]
                w = ot.sinkhorn2(a, b, M, reg=1, numitermax=10000, stopThr=1e-15).item()
                L_s[i,k,j] = time.time()-t2
            except:
                L_s[i,k,j] = np.inf

            try:
                t1 = time.time()
                ip = x0@torch.tensor(x1, dtype=torch.float, device=device).T
                M = torch.arccos(torch.clamp(ip, min=-1+1e-5, max=1-1e-5))
                a = torch.ones(x0.shape[0], device=device) / x0.shape[0]
                b = torch.ones(x1.shape[0], device=device) / x1.shape[0]
                w = ot.emd2(a, b, M).item()
                L_w[i,k,j] = time.time()-t1
            except:
                L_w[i,k,j] = np.inf

L_stswd = L_stswd[..., 1:]
L_aris3wd = L_aris3wd[..., 1:]
L_ris3wd = L_ris3wd[..., 1:]
L_s3wd = L_s3wd[..., 1:]
L_ssw2 = L_ssw2[..., 1:]
L_unif = L_unif[..., 1:]
L_ssw1 = L_ssw1[..., 1:]
L_sw2 = L_sw2[..., 1:]
L_w = L_w[..., 1:]
L_s = L_s[..., 1:]

import cycler
default_cycle = plt.rcParams['axes.prop_cycle']
plt.rcParams['axes.prop_cycle'] = cycler.Cycler((np.roll(default_cycle, -3)))

fig = plt.figure(figsize=(7.5,3.5))

for i, d in enumerate([3]):

    m_w = np.mean(L_w[i], axis=-1)
    s_w = np.std(L_w[i], axis=-1)

    plt.loglog(samples, m_w, label=r"Wasserstein", linestyle='dashdot')
    plt.fill_between(samples, m_w-s_w, m_w+s_w, alpha=0.2)


    m_s = np.mean(L_s[i], axis=-1)
    s_s = np.std(L_s[i], axis=-1)

    plt.loglog(samples, m_s, label=r"Sinkhorn", linestyle='dashdot')
    plt.fill_between(samples, m_s-s_s, m_s+s_s, alpha=0.2)

    for l, n_projs in enumerate([200]):

        m = np.mean(L_sw2[i, l], axis=-1)
        s = np.std(L_sw2[i, l], axis=-1)
        plt.plot(samples, m, label=r"$SW_2$", linewidth=1) # + r", $L=$"+str(n_projs)
        plt.fill_between(samples, m-s, m+s,alpha=0.2)

        m = np.mean(L_ssw1[i, l], axis=-1)
        s = np.std(L_ssw1[i, l], axis=-1)
        plt.plot(samples, m, label=r"$SSW_1$", linestyle='dashed', marker='s') # + r" $L=$"+str(n_projs)
        plt.fill_between(samples, m-s, m+s,alpha=0.2)

        m = np.mean(L_ssw2[i, l], axis=-1)
        s = np.std(L_ssw2[i, l], axis=-1)

        plt.plot(samples, m, label=r"$SSW_2$, BS", linestyle='dashed', marker='^') # + r" $L=$"+str(n_projs)
        plt.fill_between(samples, m-s, m+s,alpha=0.2)

        m = np.mean(L_unif[i, l], axis=-1)
        s = np.std(L_unif[i, l], axis=-1)

        plt.plot(samples, m, label=r"$SSW_2$, Unif", linestyle='dashed', marker='*') # + r" $L=$"+str(n_projs)
        plt.fill_between(samples, m-s, m+s,alpha=0.2)

        m = np.mean(L_s3wd[i, l], axis=-1)
        s = np.std(L_s3wd[i, l], axis=-1)
        plt.plot(samples, m, label=r"$S3W_2$", zorder=100, linewidth=2) # + r", $L=$"+str(n_projs)
        plt.fill_between(samples, m-s, m+s,alpha=0.2)

        m = np.mean(L_ris3wd[i, l], axis=-1)
        s = np.std(L_ris3wd[i, l], axis=-1)
        plt.plot(samples, m, label=r"$RI$-$S3W_2$", zorder=100, linewidth=2) # + r", $R=$10," + r" $L=$"+str(n_projs)
        plt.fill_between(samples, m-s, m+s,alpha=0.2)

        m = np.mean(L_aris3wd[i, l], axis=-1)
        s = np.std(L_aris3wd[i, l], axis=-1)
        plt.plot(samples, m, label=r"$ARI$-$S3W_2$", zorder=100, linewidth=2) # + r", $R=$10," + r" $L=$"+str(n_projs)
        plt.fill_between(samples, m-s, m+s,alpha=0.2)

        m = np.mean(L_stswd[i, l], axis=-1)
        s = np.std(L_stswd[i, l], axis=-1)
        plt.plot(samples, m, label=r"$STSW_2$", zorder=200, linewidth=3) # + r", $R=$10," + r" $L=$"+str(n_projs)
        plt.fill_between(samples, m-s, m+s,alpha=0.2)
        

plt.xlabel("Number of samples in each distribution", fontsize=13)
plt.ylabel("Seconds", fontsize=13)
# plt.yscale()
plt.xscale("log")
    
plt.legend(fontsize=13, bbox_to_anchor=(1,0.5), loc="center left", ncol=1)
# plt.title("Computational Time", fontsize=13)
plt.grid(True)
plt.savefig("./Runtime_Comparison.png", bbox_inches="tight")
plt.close(fig)