import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FunctionsDataset(Dataset):
    samples = None

    def __init__(self, number, t_points, t_size, t_predict_points=None, noise_sigma=0.01, const_samples=False):
        def generate_f(x_shift, x_scale, y_shift, y_scale, line_shift_angle):
            return lambda x: torch.sin(x * x_scale + x_shift) * y_scale + y_shift + line_shift_angle * x

        x_shifts = torch.randn(number) * 3.14
        x_scales = torch.rand(number) * 5
        y_shifts = torch.randn(number) * 0
        y_scales = (torch.rand(number) - 0.5) * 5
        line_shift_angles = torch.rand(number) * 2 - 1

        self.functions = [generate_f(x_shift, x_scale, y_shift, y_scale, line_shift_angle)
                          for x_shift, x_scale, y_shift, y_scale, line_shift_angle
                          in zip(x_shifts, x_scales, y_shifts, y_scales, line_shift_angles)]

        self.t_points = torch.Tensor(t_points)
        self.t_predict_points = torch.Tensor(t_predict_points) if t_predict_points is not None else None

        self.t_size = t_size
        self.noise_sigma = noise_sigma
        self.const_samples = const_samples

        if self.const_samples:
            self.samples = [self.get_value(f) for f in self.functions]

    def __getitem__(self, i):
        if self.const_samples:
            return self.samples[i]
        else:
            return self.get_value(self.functions[i])

    def __len__(self):
        return len(self.functions)

    def get_value(self, function):
        if self.t_predict_points is None:
            return self.sample_function(function, self.t_points)
        else:
            return self.sample_function(function, self.t_points) + self.sample_function(function, self.t_predict_points)

    def sample_function(self, function, times):
        X = function(times).unsqueeze(-1)
        X = X + torch.randn_like(X) * self.noise_sigma

        indices = torch.tensor(
            np.sort(np.random.choice(np.arange(X.size(0)), size=self.t_size, replace=False)),
            dtype=torch.int64,
        )
        mask = torch.zeros(X.size(0), dtype=torch.uint8)
        mask[indices] = 1

        return X, times, mask

    def get_function_values(self, i, times):
        return self.functions[i](times)


class FunctionsDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=128, shuffle=True):
        def create_tensor(sample):
            x = torch.stack([s[0] for s in sample], dim=0)
            t = sample[0][1]
            mask = torch.stack([s[2] for s in sample], dim=0)

            if len(sample[0]) == 6:
                x_to_predict = torch.stack([s[3] for s in sample], dim=0)
                t_to_predict = sample[0][4]
                mask_to_predict = torch.stack([s[5] for s in sample], dim=0)
                return x, t, mask, x_to_predict, t_to_predict, mask_to_predict

            return x, t, mask

        super(FunctionsDataLoader, self).__init__(dataset, batch_size=batch_size,
                                                  shuffle=shuffle, collate_fn=create_tensor)

    def get_functions_values(self, indices, times):
        return torch.stack([self.dataset.get_function_values(i, times) for i in indices], dim=0)
