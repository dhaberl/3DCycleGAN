import torch


class ResidualLoss(torch.nn.Module):

    def __init__(self) -> None:
        super(ResidualLoss, self).__init__()

    def forward(self, image, reference):
        residual_loss = torch.sum(torch.abs(image - reference))

        # return torch.log(residual_loss)
        return residual_loss
