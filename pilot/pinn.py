"""Neural-network pilot: u-hat is a fully connected network fit to the observations (z_i, U_i)."""
import numpy as np
import torch
from torch import nn

ACTIVATIONS = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}


class Network(nn.Module):
    """z -> mean + std f(2(z - lo)/(hi - lo) - 1), with f a fully connected network and (lo, hi) the domain per coordinate.

    The scaling is part of the model, so the network takes z in the coordinates of the domain and returns values in the
    units of U: autograd derivatives in z are those of u-hat itself, with no chain rule by hand.
    """

    def __init__(self, domain, mean, std, hidden, activation):
        super().__init__()
        lo, hi = np.array(domain, dtype=float).T
        self.register_buffer("lo", torch.as_tensor(lo))
        self.register_buffer("scale", torch.as_tensor(2/(hi - lo)))
        self.register_buffer("mean", torch.as_tensor(mean))
        self.register_buffer("std", torch.as_tensor(std))
        widths, layers = (len(domain), *hidden), []
        for a, b in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(a, b), ACTIVATIONS[activation]()]
        self.f = nn.Sequential(*layers, nn.Linear(widths[-1], len(mean)))

    def forward(self, z):
        # (N, d) points -> (N, c) values
        return self.mean + self.std*self.f((z - self.lo)*self.scale - 1)


def sobolev_norm(f, z, order):
    """sum_c sum_{|alpha| <= order} mean_j (d^alpha f_c(z_j))^2 for f from (N, d) points to (N, c) values, by autograd.

    Each multi-index alpha counts once (d_x d_t and d_t d_x are one term), as in the W^{k,2} norm. The result keeps its
    graph, so it can be backpropagated to the parameters of f.
    """
    z = z.detach().requires_grad_(True)
    out = f(z)
    total = 0
    for c in range(out.shape[1]):
        # Derivatives of order l, keyed by the non-decreasing tuple of the axes they are taken along
        level = {(): out[:, c]}
        total = total + out[:, c].square().mean()
        for _ in range(order):
            deeper = {}
            for axes, g in level.items():
                grad = torch.autograd.grad(g.sum(), z, create_graph=True)[0]
                for a in range(axes[-1] if axes else 0, z.shape[1]):
                    deeper[axes + (a,)] = grad[:, a]
            total = total + sum(g.square().mean() for g in deeper.values())
            level = deeper
    return total


def as_uhat(net, chunk=2**16):
    """The trained network as u-hat(*z): numpy coordinates of any common shape -> numpy float64 of shape (c, *shape)."""
    p = next(net.parameters())

    def uhat(*z):
        z = np.broadcast_arrays(*z)
        points = np.stack([zi.ravel() for zi in z], axis=1)
        with torch.no_grad():
            values = [net(torch.as_tensor(points[i:i + chunk], dtype=p.dtype, device=p.device)).cpu().numpy()
                      for i in range(0, len(points), chunk)]
        return np.concatenate(values).astype(np.float64).T.reshape((-1,) + z[0].shape)

    return uhat


class PINN:
    """pilot(data, seed) -> u-hat, as perturbed_truth, with u-hat a network from the d coordinates to the c components.

    The instance holds only the architecture and the training settings, so one instance in experiments.py serves every
    replication. Each call builds a new network from its own seed and the shapes of its data, so no weights are shared
    across replications.

    hidden:     widths of the hidden layers (length of the vector denotes the depth)
    activation: one of ACTIVATIONS
    epochs:     epoch
    lr:         learning rate
    kappa:      strength of the Sobolev-norm penalty kappa ||u-hat||^2 added to the mean squared error; 0 for none
    sobolev_order: k of ||u-hat||^2_{W^{k,2}} = sum_c sum_{|alpha| <= k} mean_j (d^alpha u-hat_c(z_j))^2, mixed
                derivatives included and taken in the coordinates of the domain: 6 terms per component for k = 2, d = 2
    n_collocation: uniform points on the domain, drawn afresh each epoch, at which the penalty's integral is averaged.
                The same number at every n, as in Atchade & Mukherjee (2026), Sec. 4.2
    batch_size: observations per step; None for the full batch
    device:     "cpu", "cuda", "mps", or None for cuda, then mps (only with float32), then cpu
    dtype:      "float32" or "float64"; MPS has only float32
    validation: fraction of the n observations held out at random to choose the number of epochs, at most epochs, as
                the one with the least mean squared error on them, checked after every epoch; None to train for epochs
    patience:   with validation, stop after this many epochs without a new least validation error; None to run all
    refit:      with validation, train a new network from the same seed on all n observations for the chosen number of
                epochs (True), or keep the network at that epoch, fit to the other observations only (False)
    """

    def __init__(self, hidden=(64, 64, 64), activation="tanh", epochs=5000, lr=1e-3, kappa=0.0, sobolev_order=2,
                 n_collocation=4000, batch_size=None, device=None, dtype="float64", validation=None, patience=None,
                 refit=True):
        hidden = tuple(int(h) for h in hidden)
        if not hidden or min(hidden) < 1:
            raise ValueError("hidden must be a non-empty tuple of positive widths")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {tuple(ACTIVATIONS)}")
        if int(epochs) != epochs or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= kappa < float("inf"):
            raise ValueError("kappa must be finite and non-negative")
        if int(sobolev_order) != sobolev_order or sobolev_order < 1:
            raise ValueError("sobolev_order must be a positive integer")
        if int(n_collocation) != n_collocation or n_collocation < 1:
            raise ValueError("n_collocation must be a positive integer")
        if batch_size is not None and (int(batch_size) != batch_size or batch_size < 1):
            raise ValueError("batch_size must be None or a positive integer")
        if dtype not in ("float32", "float64"):
            raise ValueError('dtype must be "float32" or "float64"')
        if device is not None and str(device).split(":")[0] == "mps" and dtype == "float64":
            raise ValueError('MPS has no float64: use dtype="float32", or device="cpu"')
        if validation is not None and not 0 < validation < 1:
            raise ValueError("validation must be None or a fraction in (0, 1)")
        if patience is not None and (int(patience) != patience or patience < 1):
            raise ValueError("patience must be None or a positive integer")
        self.hidden, self.activation = hidden, activation
        self.epochs, self.lr, self.batch_size = int(epochs), lr, batch_size
        self.kappa, self.sobolev_order, self.n_collocation = float(kappa), int(sobolev_order), int(n_collocation)
        self.device, self.dtype = device, dtype
        self.validation, self.patience, self.refit = validation, patience, bool(refit)

    def __call__(self, data, seed=None):
        # Builds and trains a network on (data.points, data.U), and returns u-hat(*z) of shape (c, *z[0].shape), with
        # the number of epochs it was trained for as u-hat.epochs
        z, U, net, batches, collocation, split = self._setup(data, seed)
        if self.validation is None:
            epochs = self._train(net, z, U, data.domain, batches, collocation, self.epochs)
        else:
            n = len(z)
            n_val = min(max(round(self.validation*n), 1), n - 1)
            order = torch.as_tensor(split.permutation(n), device=z.device)
            val, fit = order[:n_val], order[n_val:]
            epochs = self._train(net, z[fit], U[fit], data.domain, batches, collocation, self.epochs, (z[val], U[val]))
            if self.refit:
                z, U, net, batches, collocation, _ = self._setup(data, seed)
                self._train(net, z, U, data.domain, batches, collocation, epochs)
        uhat = as_uhat(net.eval().requires_grad_(False))
        uhat.epochs = epochs
        return uhat

    def _train(self, net, z, U, domain, batches, collocation, epochs, validation=None):
        # Adam on (1/n) sum_i ||U_i - u-hat(z_i)||^2 + kappa ||u-hat||^2_{W^{k,2}}, both in the units of U and the
        # coordinates of the domain. The collocation points are drawn afresh each epoch, and only when kappa > 0.
        # With validation = (z', U'), the network ends at the epoch with the least mean squared error on (z', U'), the
        # penalty left out. Returns the number of epochs the network ends at
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        n, size = len(z), min(self.batch_size or len(z), len(z))
        lo, hi = np.array(domain, dtype=float).T
        best, best_epoch, best_state = float("inf"), epochs, None
        for epoch in range(1, epochs + 1):
            order = None if size == n else torch.as_tensor(batches.permutation(n), device=z.device)
            if self.kappa > 0:
                zc = torch.as_tensor(collocation.uniform(lo, hi, (self.n_collocation, len(lo))), dtype=z.dtype, device=z.device)
            for start in range(0, n, size):
                i = slice(None) if order is None else order[start:start + size]
                loss = (net(z[i]) - U[i]).square().sum(dim=1).mean()
                if self.kappa > 0:
                    loss = loss + self.kappa*sobolev_norm(net, zc, self.sobolev_order)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if validation is not None:
                with torch.no_grad():
                    error = (net(validation[0]) - validation[1]).square().sum(dim=1).mean().item()
                if error < best:
                    best, best_epoch, best_state = error, epoch, {k: v.clone() for k, v in net.state_dict().items()}
                elif self.patience is not None and epoch - best_epoch >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return best_epoch

    def _device(self):
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available() and self.dtype == "float32":
            return torch.device("mps")
        return torch.device("cpu")

    def _setup(self, data, seed):
        # The z_i as (n, d) and the U_i as (n, c) tensors, a new network, and generators for the mini-batch order, the
        # collocation points and the validation split. seed is anything np.random.SeedSequence takes, such as main.run's
        # (seed, 1), split into four streams: the initial weights, built on the CPU so that they are the same on every
        # device; the batch order; the collocation points, so that changing kappa leaves the batch order as it was; and
        # the validation split. The first three are those of spawn(3), so runs without validation are unchanged
        device, dtype = self._device(), getattr(torch, self.dtype)
        init, batches, collocation, split = np.random.SeedSequence(seed).spawn(4)
        z = np.stack([p.ravel() for p in data.points], axis=1)
        U = data.U.reshape(len(data.U), -1).T
        std = U.std(axis=0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init.generate_state(1, np.uint64)[0]))
            net = Network(data.domain, U.mean(axis=0), np.where(std > 0, std, 1.), self.hidden, self.activation)
        tensor = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
        return (tensor(z), tensor(U), net.to(device=device, dtype=dtype), np.random.default_rng(batches),
                np.random.default_rng(collocation), np.random.default_rng(split))
