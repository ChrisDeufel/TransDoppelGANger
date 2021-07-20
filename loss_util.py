import torch
import torch.nn as nn


def gradient_penalty(disciminator, real, fake, device="cpu"):
    BATCH_SIZE, DIM = real.shape
    alpha = torch.rand((BATCH_SIZE, 1)).repeat(1, DIM).to(device)
    interpolated_data = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = disciminator(interpolated_data)

    # Take the gradient of the scores with respect to the interpolated data
    gradient = torch.autograd.grad(
        inputs=interpolated_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty