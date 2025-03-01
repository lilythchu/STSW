# Source: https://github.com/mint-vu/s3wd/blob/main/src/experiments/Density%20Estimation/train.py
import sys
import torch
import argparse
import os
from datetime import datetime

import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from tqdm.auto import trange

from datasets import EarthDataHandler
sys.path.append("../")
from utils.misc import *
from utils.nf.normalizing_flows import make_NF
from utils.func import strfdelta
from methods.s3wd import s3wd_unif, ri_s3wd_unif, ari_s3wd_unif
from methods.swd import swd as sliced_wasserstein
from methods.sswd import sswd_unif as sliced_wasserstein_sphere_unif
from methods.stswd import *

"""
Adapted from Bonet et al. 2023 (https://github.com/clbonet/spherical_sliced-wasserstein)
"""

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default="ssw", help="Which loss to use")
parser.add_argument("--dataset", type=str, default="fire", help="Which dataset to use")
parser.add_argument("--n_projs", type=int, default=1000, help="Number of projections")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
parser.add_argument("--batch_size", type=int, default=15000, help="Batch size")
parser.add_argument("--n_epochs", type=int, default=20001, help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-1, help="Learning Rate")
parser.add_argument("--n_blocks", type=int, default=48, help="Number of blocks in the NF")
parser.add_argument("--n_components", type=int, default=100, help="Number of components in the NF")
parser.add_argument("--n_try", type=int, default=5, help="Number of iterations")
parser.add_argument("--ntrees", type=int, default=1000, help="Number of trees")
parser.add_argument("--nlines", type=int, default=100, help="Number of lines")
parser.add_argument("--delta", type=float, default=0.5, help="Delta value")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--eval', action='store_true',default=False)
parser.add_argument("--wb_proj", type=str, default="density")
parser.add_argument("--eval_step", type=int, default=10)
args = parser.parse_args()

device = args.device
print(f"nt: {args.ntrees}, nl: {args.nlines}, delta: {args.delta}")

# Dataset split
config = {
    "training_size":0.7,
    "validation_size":0,
    "test_size":0.3,
    "name":args.dataset
}

# Eval metrics
def loss_kl(h, log_det):
    prior = np.log(1 / (4 * np.pi)) * torch.ones(h.shape[0], device=device)
    return -(prior+log_det).mean()

def log_likelihood(h, log_det):
    prior = np.log(1 / (4 * np.pi)) * torch.ones(h.shape[0], device=device)
    return (prior+log_det)

def get_run_name(args):
    if not args.loss.startswith("stsw"):
        return f"{args.dataset}_{args.loss}_lr{args.lr}_epochs{args.n_epochs}"

    return f"{args.dataset}_{args.loss}_nt{args.ntrees}_nl{args.nlines}_lr{args.lr}_delta{args.delta}"

if __name__ == "__main__":
    n_steps = args.n_epochs
    num_projections = args.n_projs
    L_density = np.zeros((args.n_try))

    # Create directories storing results
    if not os.path.exists("./weights"):
        os.makedirs("./weights")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    lr_str = str(args.lr).replace('.', 'e-')
    print('LR:',lr_str)
    print('E:', args.n_epochs)

    STSW_obj = torch.compile(STSWD(ntrees=args.ntrees, nlines=args.nlines, p=2, delta=args.delta, device=args.device))

    for i in range(args.n_try):
        if args.eval:
            pass
            
        handler = EarthDataHandler(config, eps=1e-5)
        train_loader, val_loader, est_loader = handler.get_dataloaders(args.batch_size, args.batch_size)

        for test_data, _ in val_loader:
            break

        model = make_NF(3, args.n_blocks, args.n_components, device=device).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of parameters: ',n_params)
        delta_train = torch.nn.Parameter(torch.tensor(2.0, requires_grad=True))
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': delta_train}
        ], lr=args.lr)
        
        if args.batch_size>len(train_loader.dataset):
            batch_size = len(train_loader.dataset) # 500
        else:
            batch_size = args.batch_size

        if args.pbar:
            pbar = trange(n_steps)
        else:
            pbar = range(n_steps)

        test_nll = []

        start_time = datetime.now()
        for k in pbar:
            for data, _ in train_loader:
                optimizer.zero_grad()

                X_target = data.to(device)
                x, log_det = model(X_target)

                if args.loss == "ssw":
                    loss = sliced_wasserstein_sphere_unif(x[-1], num_projections, device)
                elif args.loss == 's3w':
                    loss = s3wd_unif(x[-1], 2, None, num_projections, device)
                elif args.loss == "ri_s3w":
                    loss = ri_s3wd_unif(x[-1], 2, None, num_projections, 1, device)
                elif args.loss == 'ari50':
                    loss = ari_s3wd_unif(x[-1], n_projs=num_projections, p=2, n_rotations=50, pool_size=1000, device=device)
                elif args.loss == "sw":
                    z = F.normalize(torch.randn(batch_size, 3, device=device), p=2, dim=-1)
                    loss = sliced_wasserstein(x[-1], z, 2, num_projections, device)
                elif args.loss == "stsw":
                    root, intercept = generate_spherical_trees_frames(ntrees=args.ntrees, nlines=args.nlines, d=x[-1].shape[-1], device=args.device)
                    Y_unif = F.normalize(torch.randn(batch_size, 3, device=device), p=2, dim=-1)
                    loss = STSW_obj(x[-1], Y_unif, root, intercept)
                
                loss.backward()
                optimizer.step()

                if args.pbar:
                    pbar.set_postfix_str(f"loss = {loss.item():.3f}")

                z, log_det = model(test_data.to(device))
                density = log_likelihood(z[-1], log_det).detach().cpu()
                test_nll.append(-density.mean())

                if args.eval: 
                    if (k + 1) % args.eval_step == 0:
                        pass
            
        end_time = datetime.now()
        run_time = strfdelta(end_time - start_time, "{hours}:{minutes}:{seconds}")

        L_density[i] = -density.mean()
        print(k, L_density[i], flush=True)

        torch.save(model.state_dict(), f"./weights/nf_density_{get_run_name(args)}_{i}.model")
        np.savetxt(f"./results/evol_nll_{get_run_name(args)}_{i}", test_nll)
        with open(f"stats.txt", "a") as f:
            print(f"Training done in {run_time} for {get_run_name(args)}_{i}", file=f)
        
    print("Mean", np.mean(L_density), np.std(L_density), flush=True)
    with open(f"stats.txt", "a") as f:
        print("NLL: Mean", np.mean(L_density), np.std(L_density), file=f)
    np.savetxt(f"./results/NLL_{get_run_name(args)}", L_density)
