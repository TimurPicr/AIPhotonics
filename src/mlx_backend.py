"""Native MLX implementation of the existing U-Net, optics and training loss.

Training is orchestrated by src.train. PyTorch remains the checkpoint format and
the independent reference used by src.evaluate and src.global_gs.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map
import numpy as np
import torch

from src.physics import UnitaryFresnel, output_pitch_m


def _block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(math.gcd(out_channels, 4), out_channels, eps=1e-5, pytorch_compatible=True),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(math.gcd(out_channels, 4), out_channels, eps=1e-5, pytorch_compatible=True),
        nn.ReLU(),
    )


def _phase(field):
    field = mx.where(mx.abs(field) > 1e-8, field, mx.ones_like(field))
    return mx.arctan2(field.imag, field.real)


def _fft(field, *, inverse=False):
    transform = mx.fft.ifft2 if inverse else mx.fft.fft2
    return mx.fft.fftshift(transform(mx.fft.ifftshift(field, axes=(-2, -1)),
                                     axes=(-2, -1), norm='ortho'), axes=(-2, -1))


class MLXHologramUNet(nn.Module):
    """Use NHWC convolutions and the same NCHW optical fields as HologramUNet."""

    def __init__(self, reference):
        super().__init__()
        c = reference.head.in_channels
        self.encoder1 = _block(4, c)
        self.encoder2 = _block(c, 2 * c)
        self.encoder3 = _block(2 * c, 4 * c)
        self.bottleneck = _block(4 * c, 8 * c)
        self.decoder3 = _block(12 * c, 4 * c)
        self.decoder2 = _block(6 * c, 2 * c)
        self.decoder1 = _block(3 * c, c)
        self.head = nn.Conv2d(c, 4, 1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self._rounds = reference.rounds
        self._config = reference.config
        self._weights = mx.array(reference.plane_weights.cpu().numpy())
        # Reuse the reference's double-precision coefficient construction, then
        # keep these fixed complex64 coefficients on the MLX device.
        field = torch.zeros((1, 1, self._config.image_size, self._config.image_size), dtype=torch.complex64)
        operators = [UnitaryFresnel(field, z, config=self._config) for z in self._config.z_m]
        self._input_chirp = mx.stack([mx.array(op.input_chirp.numpy()) for op in operators])[None]
        self._output_factor = mx.stack([mx.array(op.output_factor.numpy()) for op in operators])[None]
        self._scales = mx.array([output_pitch_m(z, self._config.image_size, config=self._config)
                                 / self._config.slm_pitch_m for z in self._config.z_m])[None, :, None, None]
        state = reference.state_dict()
        weights = []
        for key, _ in tree_flatten(self.parameters()):
            torch_key = key.replace('.layers.', '.')
            array = state[torch_key].cpu().numpy()
            if array.ndim == 4:
                array = array.transpose(0, 2, 3, 1)
            weights.append((key, mx.array(array)))
        self.update(tree_unflatten(weights))

    def predict_plane_phases(self, targets):
        x = targets.transpose(0, 2, 3, 1)
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        deep = self.bottleneck(self.pool(e3))
        d3 = self.decoder3(mx.concatenate((self.up(deep), e3), axis=-1))
        d2 = self.decoder2(mx.concatenate((self.up(d3), e2), axis=-1))
        d1 = self.decoder1(mx.concatenate((self.up(d2), e1), axis=-1))
        return (math.pi * mx.tanh(self.head(d1))).transpose(0, 3, 1, 2)

    def __call__(self, targets):
        plane_phases = self.predict_plane_phases(targets)
        amplitudes = mx.sqrt(targets) if self._config.reconstruction_value == 'intensity' else targets
        for index in range(self._rounds):
            fields = amplitudes * mx.exp(1j * plane_phases)
            returned = _fft(fields * mx.conj(self._output_factor), inverse=True)
            returned = returned * mx.conj(self._input_chirp) * self._scales
            mask_phase = _phase(mx.sum(returned * self._weights, axis=1, keepdims=True))
            fields = _fft(mx.exp(1j * mask_phase) * self._input_chirp) * self._output_factor / self._scales
            if index + 1 < self._rounds:
                plane_phases = _phase(fields)
        recons = mx.abs(fields)
        if self._config.reconstruction_value == 'intensity':
            recons = mx.square(recons)
        return mask_phase, recons / mx.maximum(mx.max(recons, axis=(-2, -1), keepdims=True),
                                               self._config.normalization_eps)


def reconstruction_loss(recons, targets, *, alpha=0.7, beta=0.3):
    """Match src.losses, including valid SSIM windows and constant-plane rules."""
    pool = nn.AvgPool2d(7, stride=1)
    x, y = (a.transpose(0, 2, 3, 1) for a in (recons, targets))
    mean_x, mean_y = pool(x), pool(y)
    var_x = (49 / 48) * mx.maximum(pool(x * x) - mean_x * mean_x, 0)
    var_y = (49 / 48) * mx.maximum(pool(y * y) - mean_y * mean_y, 0)
    cov = (49 / 48) * (pool(x * y) - mean_x * mean_y)
    luminance = (2 * mean_x * mean_y + 0.01**2) / (mean_x * mean_x + mean_y * mean_y + 0.01**2)
    structure = (2 * cov + 0.03**2) / (var_x + var_y + 0.03**2)
    ssim = mx.mean(mx.clip(luminance * structure, -1, 1))
    x, y = (a.reshape(a.shape[0], a.shape[1], -1) for a in (recons, targets))
    valid = (mx.max(x, axis=-1) > mx.min(x, axis=-1)) & (mx.max(y, axis=-1) > mx.min(y, axis=-1))
    x, y = x - mx.mean(x, axis=-1, keepdims=True), y - mx.mean(y, axis=-1, keepdims=True)
    x = x / mx.sqrt(mx.maximum(mx.sum(x * x, axis=-1, keepdims=True), 1e-16))
    y = y / mx.sqrt(mx.maximum(mx.sum(y * y, axis=-1, keepdims=True), 1e-16))
    pearson = mx.mean(mx.where(valid, mx.clip(mx.sum(x * y, axis=-1), -1, 1), 0))
    return alpha * (1 - ssim) + beta * (1 - pearson)


class MLXBackend:
    """Small backend interface consumed by the existing epoch loop."""

    def __init__(self, reference, *, lr, seed, loss_config):
        if not mx.metal.is_available():
            raise RuntimeError('MLX Metal GPU is unavailable')
        mx.set_default_device(mx.gpu)
        mx.random.seed(seed)
        self.model = MLXHologramUNet(reference)
        self.optimizer = optim.Adam(learning_rate=lr, eps=1e-8, bias_correction=True)
        self._buffers = {'plane_weights': reference.plane_weights.cpu().clone()}
        self._loss_config = loss_config
        self._value_grad = nn.value_and_grad(
            self.model, lambda model, targets: reconstruction_loss(model(targets)[1], targets, **loss_config))
        mx.eval(self.model.parameters())

    def step(self, targets):
        self.model.train()
        inputs = mx.array(targets.cpu().numpy())
        loss, grads = self._value_grad(self.model, inputs)
        mx.eval(loss, grads)
        value = float(loss)
        if not math.isfinite(value):
            raise FloatingPointError('Non-finite MLX training loss')
        if not bool(mx.all(mx.stack([mx.all(mx.isfinite(g)) for _, g in tree_flatten(grads)]))):
            raise FloatingPointError('Non-finite MLX gradient')
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        return value

    def __call__(self, targets):
        self.model.eval()
        phase, recons = self.model(mx.array(targets.cpu().numpy()))
        mx.eval(phase, recons)
        return torch.from_numpy(np.array(phase)), torch.from_numpy(np.array(recons))

    def state_dict(self):
        result = dict(self._buffers)
        for key, value in tree_flatten(self.model.parameters()):
            array = np.array(value)
            if array.ndim == 4:
                array = array.transpose(0, 3, 1, 2)
            result[key.replace('.layers.', '.')] = torch.from_numpy(array.copy())
        return result

    def optimizer_state_dict(self):
        def convert(value):
            if not isinstance(value, mx.array):
                return value
            if value.ndim == 0:
                return value.item()
            return torch.from_numpy(np.array(value).copy())
        return tree_map(convert, self.optimizer.state)
