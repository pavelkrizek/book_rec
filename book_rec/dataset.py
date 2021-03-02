import torch


class BookDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index, :]
        target = self.targets[index]
        feature_tensor = torch.tensor(feature, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.float)

        return feature_tensor, target_tensor
