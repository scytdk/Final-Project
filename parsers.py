import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
    parser.add_argument('--batch', default=4096, type=int, help='batch size')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--hidden_dim', default=64, type=int, help='embedding size')
    parser.add_argument('--n_layers', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--dropout', default=0.0, type=float, help='rate for edge dropout')
    parser.add_argument('--reg_weight', default=1e-7, type=float, help='l2 reg weight')
    parser.add_argument('--device', default='cuda:0', type=str, help='the gpu to use')
    return parser.parse_args()