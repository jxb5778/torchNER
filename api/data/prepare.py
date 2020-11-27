
import torch

import numpy as np


class PrepareTensorData(torch.utils.data.Dataset):

    def __init__(self, X=None, y=None):

        self.X = self._to_tensor(X) if X is not None else X
        self.y = self._to_tensor(y) if y is not None else y

    def _to_tensor(self, data):

        if not torch.is_tensor(data):

            if not isinstance(data, np.ndarray):
                data = np.asarray(data)

            prepared_data = torch.from_numpy(data)

        return prepared_data

    def __len__(self):

        length = len(self.X)

        return length

    def __getitem__(self, index):

        x_val = self.X[index]
        y_val = self.y[index]

        return x_val, y_val


class PrepareTransformerData(torch.utils.data.Dataset):

    def __init__(self, X=None, y=None):

        self.X = X
        self.y = y

    def __len__(self):

        length = len(self.X)

        return length

    def __getitem__(self, index):

        x_val = self.X[index]
        y_val = self.y[index]

        return x_val, y_val
