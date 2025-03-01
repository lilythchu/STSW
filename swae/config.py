import argparse

class Config:
    type = 'swae'
    batch_size = 500
    lr = 1e-3
    epochs = 100
    d = 3
    beta = 1
    device = 'cuda'  
    loss1 = 'BCE'
    loss2 = 'stsw'
    dataset = 'CIFAR10'
    prior = 'vmf'
    ntrees = 200
    nlines = 10
    delta = 2
    n_projs = 100
    seed = 123


def parse_args():
    parser = argparse.ArgumentParser(description='training configs')
    parser.add_argument('--type', type=str, default='swae', choices=['ae', 'swae'], help='which ae?')
    parser.add_argument('--loss1', type=str, default='BCE', help='loss1')
    parser.add_argument('--loss2', type=str, default='stsw', help='loss2')
    parser.add_argument('--beta', type=float, default=1, help='regularization coefficient')
    parser.add_argument('--d', type=int, default=3, help='embedding dim')
    parser.add_argument('--dataset', type=str, default='c10', choices=['mnist', 'c10'], help='dataset')
    parser.add_argument('--prior', type=str, default='vmf', choices=['uniform', 'vmf'], help='prior')
    parser.add_argument('--device', type=str, default="cuda:1", help='device')
    parser.add_argument('--lr', type=float, default=1e-3  , help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--ntrees', type=int, default=200)
    parser.add_argument('--nlines', type=int, default=10)
    parser.add_argument('--delta', type=float, default=2)
    parser.add_argument('--n_projs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    
    args = parser.parse_args()
    return args