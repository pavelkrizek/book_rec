import torch
import torch.nn as nn


class BookModel(nn.Module):
    """
    Simple matrix factorization BookModel
    """

    def __init__(self, n_users, n_books, n_embed, y_low=1, y_high=10.5):
        super(BookModel, self).__init__()

        self.user_embed = Embedding(n_users, n_embed)
        self.user_bias = Embedding(n_users, 1)
        self.book_embed = Embedding(n_books, n_embed)
        self.book_bias = Embedding(n_books, 1)
        self.y_low = y_low
        self.y_high = y_high

    def forward(self, x):
        users = self.user_embed(x[:, 0])
        movies = self.book_embed(x[:, 1])
        result = (users * movies).sum(dim=1, keepdim=True)
        result += self.user_bias(x[:, 0]) + self.book_bias(x[:, 1])
        y = torch.sigmoid(result) * (self.y_high - self.y_low) + self.y_low
        return y


class Embedding(nn.Embedding):
    """
    Embeddings with truncated normal inicialization for better convergence
    """

    def __init__(self, ni, nf):
        super().__init__(ni, nf)
        truncated_normal_(self.weight.data, std=0.01)


def truncated_normal_(x, mean=0.0, std=1.0):
    return x.normal_().fmod_(2).mul_(std).add_(mean)
