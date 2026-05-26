from dataclasses import dataclass


@dataclass(frozen=True)
class LagrangianConfig:
    h_y: float = 0.1
    alpha: float = 1.0
    n_quad: int = 9


@dataclass(frozen=True)
class OptimConfig:
    lr_g: float = 1e-4
    lr_T: float = 1e-4
    lr_S: float = 1e-4
    lr_G: float = 5e-4
    grad_clip: float = 0.25


@dataclass(frozen=True)
class EntropicConfig:
    epsilon: float = 1e-2


@dataclass(frozen=True)
class LBFGSInnerConfig:
    maxiter: int = 20
    memory_size: int = 10
    y1_bound: float = 10.0


@dataclass(frozen=True)
class TrainConfig:
    n_outer: int = 400
    n_inner: int = 10
    batch_size: int = 64
    method: str = "lbfgs"
    seed: int = 5
    log_every: int = 50
