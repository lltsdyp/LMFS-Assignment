"""Language-model training: AdamW, learning rate, and gradient clipping."""

from collections.abc import Iterable
import math

import torch
from torch import Tensor

from src.utils.tensors import check_same_device


@torch.no_grad()
def _adamw_update_(
    parameter: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> None:
    """Update a parameter in place; AdamW manages state and step counts."""
    beta1, beta2 = betas
    grad = grad.to(exp_avg.dtype)
    parameter.mul_(1 - lr * weight_decay)
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).add_(grad.square(), alpha=1 - beta2)
    adjusted_lr = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)
    parameter.add_(exp_avg / (exp_avg_sq.sqrt() + eps), alpha=-adjusted_lr)


@torch.no_grad()
def _clip_gradients_(
    gradients: list[Tensor], max_norm: float, eps: float = 1e-6
) -> Tensor:
    """Clip nonempty same-device gradients; return the original L2 norm."""
    check_same_device(*gradients)
    dtype = (
        torch.float64
        if any(g.dtype == torch.float64 for g in gradients)
        else torch.float32
    )
    total = torch.zeros((), device=gradients[0].device, dtype=dtype)
    for gradient in gradients:
        if gradient.is_sparse:
            raise ValueError("gradient clipping requires dense gradients")
        total += gradient.to(dtype).square().sum()
    norm = total.sqrt()
    if not torch.isfinite(norm):
        raise ValueError("global gradient norm is nonfinite")
    # Keep the clipping threshold decision on device. Nonfinite checking
    # above requires a host synchronization when gradients are on CUDA.
    factor = torch.where(norm > max_norm, max_norm / (norm + eps), 1.0)
    for gradient in gradients:
        gradient.mul_(factor.to(gradient.dtype))
    return norm


class AdamW(torch.optim.Optimizer):
    """Handwritten AdamW following CS336's uncorrected-moment denominator.

    lr_t = lr * sqrt(1 - beta2**t) / (1 - beta1**t), followed by
    p -= lr_t * m / (sqrt(v) + eps). Weight decay is applied before this
    update, with the original lr. This eps convention differs slightly
    from PyTorch AdamW. Only dense real floating-point parameters are
    supported. Half/bfloat16 moments are accumulated in float32. The
    base class provides state_dict and zero_grad.
    """

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )
        # Check final group values once, including any per-group overrides.
        # Invalid beta/epsilon values break the update's divisions; nonfinite
        # hyperparameters would corrupt parameters and moment state.
        for group in self.param_groups:
            lr, betas, eps, weight_decay = (
                group["lr"],
                group["betas"],
                group["eps"],
                group["weight_decay"],
            )
            if not math.isfinite(lr) or lr < 0:
                raise ValueError("lr must be finite and nonnegative")
            if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
                raise ValueError("betas must contain two values in [0, 1)")
            if not math.isfinite(eps) or eps <= 0:
                raise ValueError("eps must be finite and positive")
            if not math.isfinite(weight_decay) or weight_decay < 0:
                raise ValueError("weight_decay must be finite and nonnegative")

    def load_state_dict(self, state_dict):
        """Restore float32 moments without casting them through half precision."""
        super().load_state_dict(state_dict)
        for saved_group, group in zip(
            state_dict["param_groups"], self.param_groups
        ):
            for key, parameter in zip(saved_group["params"], group["params"]):
                saved = state_dict["state"].get(key, {})
                dtype = (
                    torch.float32
                    if parameter.dtype in (torch.float16, torch.bfloat16)
                    else parameter.dtype
                )
                for name in ("exp_avg", "exp_avg_sq"):
                    if name in saved:
                        self.state[parameter][name] = (
                            saved[name]
                            .to(device=parameter.device, dtype=dtype)
                            .clone()
                        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, eps, decay = group["lr"], group["eps"], group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse or not parameter.is_floating_point():
                    raise ValueError(
                        "AdamW requires dense real floating tensors"
                    )
                moment_dtype = (
                    torch.float32
                    if parameter.dtype in (torch.float16, torch.bfloat16)
                    else parameter.dtype
                )
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter, dtype=moment_dtype
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, dtype=moment_dtype
                    )
                state["step"] += 1
                _adamw_update_(
                    parameter,
                    grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    step=state["step"],
                    lr=lr,
                    betas=group["betas"],
                    eps=eps,
                    weight_decay=decay,
                )
        return loss


def cosine_learning_rate(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    cosine_steps: int,
) -> float:
    """CS336 warmup + cosine LR; cosine_steps is the final absolute step.

    At step 0 the LR is zero when warmup_steps > 0; it reaches max_lr
    at warmup_steps and min_lr at cosine_steps, staying there afterwards.
    With warmup_steps == 0, step 0 uses max_lr.
    """
    if step < 0 or warmup_steps < 0 or cosine_steps <= warmup_steps:
        raise ValueError(
            "require step >= 0 and 0 <= warmup_steps < cosine_steps"
        )
    if not (math.isfinite(max_lr) and math.isfinite(min_lr)):
        raise ValueError("learning rates must be finite")
    if not 0 <= min_lr <= max_lr:
        raise ValueError("require 0 <= min_lr <= max_lr")
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > cosine_steps:
        return min_lr
    progress = (step - warmup_steps) / (cosine_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


@torch.no_grad()
def clip_grad_norm_(
    parameters: Iterable[Tensor], max_norm: float, eps: float = 1e-6
) -> Tensor:
    """Clip the global L2 gradient norm in place, returning the original norm.

    Ignores parameters with no gradients and handles iterable/generator
    inputs. Accumulates half/bfloat16 gradients in float32. Nonfinite
    gradients raise before modification, rather than silently corrupting
    the parameters. An empty collection returns a CPU scalar zero.
    """
    if (
        not math.isfinite(max_norm)
        or max_norm < 0
        or not math.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("max_norm must be finite/nonnegative; eps positive")
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return torch.tensor(0.0)
    return _clip_gradients_(gradients, max_norm, eps)
