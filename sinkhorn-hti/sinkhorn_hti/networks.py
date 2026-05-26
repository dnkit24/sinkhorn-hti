from typing import Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from .lagrangian import assemble_metric


class FiLMMLP(eqx.Module):
    first: eqx.nn.Linear
    rest: tuple[eqx.nn.Linear, ...]
    film_scale: eqx.nn.Linear
    film_shift: eqx.nn.Linear
    activation: Callable[[jnp.ndarray], jnp.ndarray] = eqx.field(static=True)
    film_size: int = eqx.field(static=True)
    film_mode: str = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_sizes: tuple[int, ...],
        x_size: int,
        film_size: int,
        key: jax.Array,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
        film_mode: Literal["partial", "full"] = "full",
    ):
        if film_mode == "full":
            film_size = hidden_sizes[0]
        k_first, k_rest, k_scale, k_shift = random.split(key, 4)
        self.first = eqx.nn.Linear(in_size, hidden_sizes[0], key=k_first)
        rest_keys = random.split(k_rest, len(hidden_sizes))
        sizes = [hidden_sizes[0], *hidden_sizes[1:], out_size]
        self.rest = tuple(
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=rest_keys[i])
            for i in range(len(sizes) - 1)
        )
        self.film_scale = eqx.nn.Linear(x_size, film_size, key=k_scale)
        self.film_shift = eqx.nn.Linear(x_size, film_size, key=k_shift)
        self.activation = activation
        self.film_size = film_size
        self.film_mode = film_mode

    def __call__(self, z, x):
        h = self.first(z)
        gamma = self.film_scale(x)
        beta = self.film_shift(x)
        if self.film_mode == "full":
            h = gamma * h + beta
        else:
            h = h.at[: self.film_size].set(gamma * h[: self.film_size] + beta)
        h = self.activation(h)
        for layer in self.rest[:-1]:
            h = self.activation(layer(h))
        return self.rest[-1](h)


def _zero_last(mlp):
    last = mlp.rest[-1]
    zeroed = eqx.tree_at(
        lambda L: (L.weight, L.bias),
        last,
        (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)),
    )
    return eqx.tree_at(lambda m: m.rest, mlp, mlp.rest[:-1] + (zeroed,))


class PotentialNet(eqx.Module):
    mlp: FiLMMLP

    def __init__(self, dy, dx, hidden_sizes, film_size, key, film_mode="full"):
        self.mlp = FiLMMLP(
            in_size=dy, out_size=1, hidden_sizes=hidden_sizes,
            x_size=dx, film_size=film_size, key=key, film_mode=film_mode,
        )

    def __call__(self, y, x):
        return self.mlp(y, x)[0]


class MapNet(eqx.Module):
    mlp: FiLMMLP

    def __init__(self, dy, dx, hidden_sizes, film_size, key, film_mode="full"):
        self.mlp = _zero_last(FiLMMLP(
            in_size=dy, out_size=dy, hidden_sizes=hidden_sizes,
            x_size=dx, film_size=film_size, key=key, film_mode=film_mode,
        ))

    def __call__(self, y, x):
        return y + self.mlp(y, x)


class SplineNet(eqx.Module):
    mlp: FiLMMLP
    n_interior: int = eqx.field(static=True)
    dy: int = eqx.field(static=True)

    def __init__(self, dy, dx, hidden_sizes, film_size, n_interior, key, film_mode="full"):
        self.n_interior = n_interior
        self.dy = dy
        self.mlp = _zero_last(FiLMMLP(
            in_size=2 * dy, out_size=n_interior * dy, hidden_sizes=hidden_sizes,
            x_size=dx, film_size=film_size, key=key, film_mode=film_mode,
        ))

    def __call__(self, y0, y1, x):
        deltas = self.mlp(jnp.concatenate([y0, y1], axis=-1), x).reshape(self.n_interior, self.dy)
        frac = jnp.arange(1, self.n_interior + 1, dtype=jnp.float32) / (self.n_interior + 1)
        interior = (1.0 - frac)[:, None] * y0 + frac[:, None] * y1 + deltas
        return jnp.concatenate([y0[None, :], interior, y1[None, :]], axis=0)


class MetricNet(eqx.Module):
    mlp: FiLMMLP
    dy: int = eqx.field(static=True)
    n_angles: int = eqx.field(static=True)
    eigenvalue_budget: float = eqx.field(static=True)
    eigenvalue_floor: float = eqx.field(static=True)

    def __init__(
        self,
        dy,
        dx,
        hidden_sizes,
        film_size,
        eigenvalue_budget,
        key,
        eigenvalue_floor=0.3,
        film_mode="full",
    ):
        self.dy = dy
        self.n_angles = dy * (dy - 1) // 2
        self.eigenvalue_budget = eigenvalue_budget
        self.eigenvalue_floor = eigenvalue_floor
        self.mlp = FiLMMLP(
            in_size=dy, out_size=self.n_angles + dy, hidden_sizes=hidden_sizes,
            x_size=dx, film_size=film_size, key=key, film_mode=film_mode,
        )

    def __call__(self, q, x):
        out = self.mlp(q, x)
        angles, logits = out[: self.n_angles], out[self.n_angles :]
        spread = self.eigenvalue_budget - self.dy * self.eigenvalue_floor
        # floor + fixed total: smallest eigenvalue >= floor, sum of eigenvalues = budget
        eigs = self.eigenvalue_floor + spread * jax.nn.softmax(logits)
        return assemble_metric(angles, eigs, d=self.dy)


class HTIComponents(eqx.Module):
    potentials: tuple[PotentialNet, ...]
    maps: tuple[MapNet, ...]
    spline: SplineNet
    metric: MetricNet

    @classmethod
    def init(
        cls,
        key,
        n_intervals,
        dy,
        dx,
        potential_hidden=(64, 64, 64, 64),
        map_hidden=(64, 64, 64, 64),
        spline_hidden=(512, 512),
        metric_hidden=(128, 128),
        film_size=16,
        spline_knots=15,
        eigenvalue_budget=2.0,
        eigenvalue_floor=0.3,
        film_mode="full",
    ):
        keys = random.split(key, 2 + 2 * n_intervals)
        gs = tuple(
            PotentialNet(dy, dx, potential_hidden, film_size, keys[2 * k], film_mode=film_mode)
            for k in range(n_intervals)
        )
        Ts = tuple(
            MapNet(dy, dx, map_hidden, film_size, keys[2 * k + 1], film_mode=film_mode)
            for k in range(n_intervals)
        )
        S = SplineNet(
            dy, dx, spline_hidden, film_size,
            n_interior=max(spline_knots - 2, 1),
            key=keys[-2], film_mode=film_mode,
        )
        G = MetricNet(
            dy, dx, metric_hidden, film_size,
            eigenvalue_budget=eigenvalue_budget,
            key=keys[-1],
            eigenvalue_floor=eigenvalue_floor,
            film_mode=film_mode,
        )
        return cls(potentials=gs, maps=Ts, spline=S, metric=G)
