import torch
import torch.nn as nn


class DLOSS(nn.Module):
    def __init__(self, loss_name="KL", clamp_min=-10.0, clamp_max=10.0):
        super().__init__()
        self.loss_name = loss_name
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        if loss_name == "BCE":
            self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, u, v):
        # u = torch.clamp(u, self.clamp_min, self.clamp_max)
        # v = torch.clamp(v, self.clamp_min, self.clamp_max)

        if self.loss_name == "KL":
            return -(torch.mean(u) - torch.mean(torch.exp((v - 1))))

        elif self.loss_name == "RKL":
            return -(torch.mean(-torch.exp(u)) - torch.mean(-1 - v))

        elif self.loss_name == "JS":
            two = torch.tensor(2.0, device=u.device, dtype=u.dtype)
            return -(
                torch.mean(two - (1 + torch.exp(-u)))
                - torch.mean(-(two - torch.exp(v)))
            )

        elif self.loss_name == "Pearson":
            return -(torch.mean(u) - 0.25 * v**2 + v)

        elif self.loss_name == "BCE":
            labels_real = torch.ones_like(u)
            loss_real = self.bce_loss(u, labels_real)
            labels_fake = torch.zeros_like(v)
            loss_fake = self.bce_loss(v, labels_fake)
            return loss_real + loss_fake

        else:
            raise NotImplementedError(f"Unknown divergence type: {self.loss_name}")


class GLOSS(nn.Module):
    def __init__(self, loss_name="KL", clamp_min=-10.0, clamp_max=10.0):
        super().__init__()
        self.loss_name = loss_name
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        if loss_name == "BCE":
            self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, v):
        # v = torch.clamp(v, min=self.clamp_min, max=self.clamp_max)

        two = torch.tensor(2.0, device=v.device, dtype=v.dtype)
        if self.loss_name == "KL":
            return -torch.mean(torch.exp((v - 1)))

        elif self.loss_name == "RKL":
            return -torch.mean(-1 - v)

        elif self.loss_name == "JS":
            return -torch.mean(-(two - torch.exp(v)))

        elif self.loss_name == "Pearson":
            return -torch.mean(0.25 * v**2 + v)

        elif self.loss_name == "BCE":
            labels_real = torch.ones_like(v)
            loss_G = self.bce_loss(v, labels_real)
            return loss_G

        else:
            raise NotImplementedError(f"Unknown divergence type: {self.loss_name}")
