import argparse
import numpy as np
import torch
import warnings
from libs.functions_dataset import FunctionsDataset, FunctionsDataLoader
from libs.latent_ode import LatentODE
from libs.trainer import Trainer


def main():
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='latent_ode')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_epoch', default=100, type=int)
    parser.add_argument('--version', default=1, type=int)
    parser.add_argument('--train_dataset_size', default=1000, type=int)
    parser.add_argument('--val_dataset_size', type=int)
    parser.add_argument('--time_size', default=30, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--log_dir', default='./logs', type=str)

    parser.add_argument('--save', dest='save', action='store_true')
    parser.set_defaults(save=False)
    parser.add_argument('--const_samples', dest='const_samples', action='store_true')
    parser.set_defaults(const_samples=False)

    args = parser.parse_args()

    device = torch.device(args.device)
    model = LatentODE(x_size=1, z_size=64, hid_size=128, noise_sigma=0.01, device=device, reverse=False,
                      method='dopri5')

    train_functions = FunctionsDataset(args.train_dataset_size, np.linspace(0, 5, 100), args.time_size,
                                       t_predict_points=np.linspace(5, 10, 100)[1:], const_samples=args.const_samples)
    train_dataloader = FunctionsDataLoader(train_functions, batch_size=args.batch_size)

    val_dataset_size = args.val_dataset_size if args.val_dataset_size is not None else args.train_dataset_size
    val_functions = FunctionsDataset(val_dataset_size, np.linspace(0, 5, 100), args.time_size,
                                     t_predict_points=np.linspace(5, 10, 100)[1:], const_samples=args.const_samples)
    val_dataloader = FunctionsDataLoader(val_functions, batch_size=args.batch_size, shuffle=False)

    trainer = Trainer(model, args.model, train_dataloader, val_dataloader, device=device, version=args.version,
                      log_dir=args.log_dir, learning_rate=args.learning_rate, custom_time=np.linspace(5, 30, 600))
    trainer.train(args.n_epoch, args.save)


if __name__ == '__main__':
    main()
