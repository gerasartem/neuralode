import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from libs.utils import init_mask


class Trainer:
    def __init__(self, model, model_name, train_dataloader, val_dataloader=None, mode='predict_same',
                 device=None, version=1, log_dir='./logs', learning_rate=1e-3, custom_time=None):
        self.model_name = model_name
        self.version = version

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.mode = mode

        self.device = device
        if self.device is None:
            self.device = torch.device('cpu')

        self.custom_time = custom_time
        if self.custom_time is not None:
            self.custom_time = torch.Tensor(custom_time).to(self.device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.logger = SummaryWriter(log_dir=log_dir)

    def process_epoch(self, dataloader, train=True):
        losses, mses, maes = [], [], []
        self.model.train(train)
        for batch in dataloader:
            x = batch[0].to(self.device)
            t = batch[1].to(self.device)
            mask = batch[2].to(self.device)

            x_to_predict, t_to_predict, mask_to_predict = None, None, None
            if len(batch) == 6:
                x_to_predict = batch[3].to(self.device)
                t_to_predict = batch[4].to(self.device)
                mask_to_predict = batch[5].to(self.device)

            loss, mse, mae = self.model.compute_loss(x, t, mask, x_to_predict, t_to_predict, mask_to_predict,
                                                     return_metrics=True)
            if train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            losses.append(float(loss.cpu().detach()))
            mses.append(float(mse.cpu().detach()))
            maes.append(float(mae.cpu().detach()))

        return np.mean(losses), np.mean(mses), np.mean(maes)

    def eval_single_batch(self, dataloader, custom_time=None, expand_batch=False):
        x, t, mask, pred, t_pred, mask_pred = (None, ) * 6

        self.model.eval()
        for batch in dataloader:
            x = batch[0].to(self.device)
            t = batch[1].to(self.device)
            mask = batch[2].to(self.device)

            if expand_batch:
                custom_time = batch[4].to(self.device)

            with torch.no_grad():
                pred = self.model.sample_similar(x, t, mask, custom_time=custom_time)
            break

        if expand_batch:
            x = torch.cat([batch[0], batch[3]], dim=1)
            t = torch.cat([batch[1], batch[4]], dim=0)
            mask = torch.cat([batch[2], batch[5]], dim=1)
            mask_pred = batch[5]

        t_pred = custom_time if custom_time is not None else t

        return x.cpu(), t.cpu(), mask.cpu(), pred.cpu(), t_pred.cpu(), mask_pred

    def draw_plot(self, epoch, x_real, t_real, x_pred, t_pred=None,
                  mask_real=None, mask_pred=None, mode_real='scatter', mode_pred='plot',
                  figure_name='Val/img', vline_shift=None):
        if t_pred is None:
            t_pred = t_real
        if mask_real is None:
            mask_real = init_mask(x_real)
        if mask_pred is None:
            mask_pred = init_mask(x_pred)

        def add_plot(X, t, mask, mode, marker='o', ls='-'):
            for x, m, color in zip(X, mask, ['r', 'g', 'b']):
                if mode == 'scatter':
                    plt.scatter(t[m].cpu().detach().numpy().ravel(), x[m].cpu().detach().numpy().ravel(),
                                color=color, marker=marker)
                elif mode == 'plot':
                    plt.plot(t[m].cpu().detach().numpy().ravel(), x[m].cpu().detach().numpy().ravel(),
                             color=color, ls=ls)
                else:
                    raise 'Unknown plot mode {}'.format(mode)

        figure = plt.figure(figsize=(8, 8))

        add_plot(x_real, t_real, mask_real, mode_real, marker='o', ls='-')
        add_plot(x_pred, t_pred, mask_pred, mode_pred, marker='x', ls='--')

        if vline_shift is not None:
            plt.vlines(vline_shift,
                       min(x_real.min(), x_pred.min()).cpu() - 1,
                       max(x_real.max(), x_pred.max()).cpu() + 1)

        plt.grid()
        plt.ylim((min(x_real.min(), x_pred.min()).cpu() - 1, max(x_real.max(), x_pred.max()).cpu() + 1))
        self.logger.add_figure(figure_name, figure, global_step=epoch)
        plt.close()

    def log_metrics(self, loss, mse, mae, epoch, type='Train'):
        if loss is not None:
            self.logger.add_scalar(type + '/Loss', np.mean(loss), epoch)
        if mse is not None:
            self.logger.add_scalar(type + '/MSE', np.mean(mse), epoch)
        if mae is not None:
            self.logger.add_scalar(type + '/MAE', np.mean(mae), epoch)

    def train(self, epoch_num, save=True):
        for epoch in tqdm(range(epoch_num)):
            loss, mse, mae = self.process_epoch(self.train_dataloader, train=True)
            self.log_metrics(loss, mse, mae, epoch, type='Train')

            if (epoch + 1) % 20 == 0:
                x, t, mask, pred, t_pred, mask_pred = self.eval_single_batch(dataloader=self.train_dataloader,
                                                                             expand_batch=True)
                self.draw_plot(epoch, x, t, pred, t_pred=t_pred, mask_real=mask, mask_pred=mask_pred,
                               mode_real='scatter', mode_pred='scatter', figure_name='Train/img',
                               vline_shift=5.0)

            if self.val_dataloader is not None:
                loss, mse, mae = self.process_epoch(self.val_dataloader, train=False)
                self.log_metrics(loss, mse, mae, epoch, type='Val')

                if (epoch + 1) % 20 == 0:
                    pred = self.eval_single_batch(dataloader=self.val_dataloader, custom_time=self.custom_time)[3]
                    x = self.val_dataloader.get_functions_values([0, 1, 2], self.custom_time.cpu())
                    self.draw_plot(epoch, x, self.custom_time, pred, mode_real='plot', mode_pred='plot',
                                   figure_name='Val/img', vline_shift=[5.0, 10.0])

        if save:
            self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_name + '_v{}.pt'.format(self.version))

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_name + '_v{}.pt'.format(self.version)))
        self.model.eval()
