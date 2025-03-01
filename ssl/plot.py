import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
import torchvision
from torch import nn
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)

class ResNet(nn.Module):
    def __init__(self, in_channel: int = 3, feat_dim: int = 128, no_bias = False):
        super().__init__()
        self.rn = resnet18(num_classes=32 * 32)

        if no_bias:
            self.rn.fc = nn.Linear(*self.rn.fc.weight.data.shape[::-1], bias=False)
        self.rn.maxpool = nn.Identity()
        self.rn.conv1 = nn.Conv2d(in_channel, 64,
                kernel_size=3, stride=1, padding=2, bias=False)

        self.predictor = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(32 * 32, feat_dim, bias=False),
            L2Norm(),
        )

    def forward(self, x, layer_index:int = -1):
        if layer_index == -1:
            return self.predictor(self.rn(x))

        if layer_index == -2:
            return self.rn(x)

def get_embeddings(encoder):
    all_z = None
    all_y = None

    n_batches = 100
    for x, y in tqdm(limited(dataloader, size=n_batches)):
        with torch.no_grad():
            z = encoder(x.cuda(), layer_index = -1)

        if all_z is None:
            all_z = z.cpu()
        else:
            all_z = torch.cat((all_z, z.cpu()))

        if all_y is None:
            all_y = y.cpu()
        else:
            all_y = torch.cat((all_y, y.cpu()))
    return all_z, all_y

def scatter_dists(all_z, all_y, desc=None, include_legend=False, export_legend=False, fig_name=None):
    fig = plt.figure()
    plt.subplot(111, projection="mollweide")

    for i in range(10):
        selector = all_y == i
        θ = torch.atan2(-all_z[selector, 1], -all_z[selector, 0])
        ϕ = torch.asin(all_z[selector,2])
        plt.scatter(θ, ϕ,
                    s=.7, label=cifar10.classes[i]) # marker=',', label = "")

    # desc is None or plt.title(desc)
    if include_legend:
        legend = plt.legend(bbox_to_anchor=(1,0.5), loc="center left", markerscale=15, fontsize=16)
        # legend = plt.legend( loc="lower center", markerscale=15, ncols=5, bbox_to_anchor=(0.5,-0.4), fontsize=12) # 1.1
    
        if export_legend:
            fig = legend.figure
            fig.canvas.draw()
            bbox  = legend.get_window_extent()
            bbox = bbox.from_extents(*(bbox.extents + np.array([-5,-5,5,5])))
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            fig.savefig('features_legend.pdf', dpi=1000, bbox_inches=bbox)


    plt.subplots_adjust(right=0.75)
    plt.grid()
    
    suffix = "_".join(desc.lower().split())
    plt.savefig(f"figures/ssl_{fig_name if fig_name else suffix}.png", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    cpt = input("Check point:")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs('figures', exist_ok=True)
    feat_dim = 3
    get_transform = lambda mean, std, resize, crop_size: torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(resize),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std,),
        ]
    )

    transform = get_transform(
        mean=(0.4915, 0.4822, 0.4466),
        std=(0.2470, 0.2435, 0.2616),
        crop_size=32,
        resize=32,
    )

    cifar10 = CIFAR10("./data", train=False, download=True, transform=transform)
    dataloader = DataLoader(cifar10, batch_size=256)
    limited = lambda gen, size=-1: (gen) if size == -1 or size > len(gen) else (x for _, x in zip(range(size), gen))

    stsw = ResNet(feat_dim=feat_dim).cuda().eval()
    checkpoint = torch.load(f"{cpt}/encoder.pth")
    stsw.load_state_dict(checkpoint)
    all_z_stsw, all_y_stsw = get_embeddings(stsw)
    scatter_dists(all_z_stsw, all_y_stsw, "STSW")

    # s3w = ResNet(feat_dim=feat_dim).cuda().eval()
    # checkpoint = torch.load("./results/method_s3w_unif_w_0.1_align_w_1.0_feat_dim_3__n_projs_200_n_rots_5_pool_size_100_epochs_200_batch_size_512_lr_0.05_momentum_0.9_seed_0_weight_decay_0.001_runtime2/encoder.pth")
    # s3w.load_state_dict(checkpoint)
    # all_z_s3w, all_y_s3w = get_embeddings(s3w)
    # scatter_dists(all_z_s3w, all_y_s3w, "S3W", include_legend=False)


    # ri_s3w = ResNet(feat_dim=feat_dim).cuda().eval()
    # checkpoint = torch.load("./results/method_ri_s3w_unif_w_0.1_align_w_1.0_feat_dim_3__n_projs_200_n_rots_5_pool_size_100_epochs_200_batch_size_512_lr_0.05_momentum_0.9_seed_0_weight_decay_0.001_runtime2/encoder.pth")
    # ri_s3w.load_state_dict(checkpoint)
    # all_z_ri_s3w, all_y_ri_s3w = get_embeddings(ri_s3w)
    # scatter_dists(all_z_ri_s3w, all_y_ri_s3w, "RI-S3W")


    # ari_s3w = ResNet(feat_dim=feat_dim).cuda().eval()
    # checkpoint = torch.load("./results/method_ari_s3w_unif_w_0.1_align_w_1.0_feat_dim_3__n_projs_200_n_rots_5_pool_size_100_epochs_200_batch_size_512_lr_0.05_momentum_0.9_seed_0_weight_decay_0.001_runtime2/encoder.pth", map_location=device)
    # ari_s3w.load_state_dict(checkpoint)
    # all_z_ari_s3w, all_y_ari_s3w = get_embeddings(ari_s3w)
    # scatter_dists(all_z_ari_s3w, all_y_ari_s3w, "ARI-S3W")


    # ssw = ResNet(feat_dim = 3).cuda().eval()
    # checkpoint = torch.load("./results/method_ssw_unif_w_20.0_align_w_1.0_feat_dim_3__n_projs_200_n_rots_5_pool_size_100_epochs_200_batch_size_512_lr_0.05_momentum_0.9_seed_0_weight_decay_0.001_runtime2/encoder.pth")
    # ssw.load_state_dict(checkpoint)
    # all_z_ssw, all_y_ssw = get_embeddings(ssw)
    # scatter_dists(all_z_ssw, all_y_ssw, "SSW")


    # sw = ResNet(feat_dim = 3).cuda().eval()
    # # SSL/results/method_sw_epochs_200_feat_dim_3_batch_size_512_num_projections_200_num_rotations_1_unif_w_1.0_align_w_1.0_lr_0.05_momentum_0.9_seed_0_weight_decay_0.001_sw_updated/encoder.pth
    # checkpoint = torch.load(f"./results/{cpt}/encoder.pth")
    # sw.load_state_dict(checkpoint)
    # all_z_sw, all_y_sw = get_embeddings(sw)
    # scatter_dists(all_z_sw, all_y_sw, "SW")


    # hypersphere = ResNet(feat_dim=feat_dim).cuda().eval()
    # checkpoint = torch.load(f"./results/{cpt}/encoder.pth")
    # hypersphere.load_state_dict(checkpoint)
    # all_z_hypersphere, all_y_hypersphere = get_embeddings(hypersphere)
    # scatter_dists(all_z_hypersphere, all_y_hypersphere, "Wang")

    # simclr = ResNet(feat_dim=feat_dim).cuda().eval()
    # checkpoint = torch.load(f"./results/{cpt}/encoder.pth")
    # simclr.load_state_dict(checkpoint)
    # all_z_simclr, all_y_simclr = get_embeddings(simclr)
    # scatter_dists(all_z_simclr, all_y_simclr, "SimCLR")


    # supervised = ResNet(feat_dim=feat_dim).cuda().eval()
    # checkpoint = torch.load(f"{cpt}/encoder.pth")
    # supervised.load_state_dict(checkpoint)
    # all_z_supervised, all_y_supervised = get_embeddings(supervised)
    # scatter_dists(all_z_supervised, all_y_supervised, "Supervised predictive")


    # random_encoder = ResNet(feat_dim=feat_dim, no_bias=True).cuda().eval()
    # all_z_rand, all_y_rand = get_embeddings(random_encoder)
    # scatter_dists(all_z_rand, all_y_rand, "Random initialization")


