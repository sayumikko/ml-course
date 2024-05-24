import math
from typing import Dict, Any, Tuple, Optional

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from weight_decay import WeightDecay


class Adam(Optimizer):

    def __init__(self, params,
                 lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-16,
                 weight_decay: float = 0,
                 optimized_update: bool = True,
                 defaults: Optional[Dict[str, Any]] = None):

        defaults = {} if defaults is None else defaults
        weight_decay = WeightDecay(weight_decay=weight_decay)
        defaults.update(weight_decay.defaults())

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults.update(dict(lr=lr, betas=betas, eps=eps))
        super().__init__(params, defaults)

        self.weight_decay = weight_decay
        self.optimized_update = optimized_update

    def init_state(self, state: Dict[str, any], group: Dict[str, any], param: nn.Parameter):
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

    def get_mv(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor):
        beta1, beta2 = group['betas']

        m, v = state['exp_avg'], state['exp_avg_sq']
    
        m.mul_(beta1).add_(grad, alpha=1 - beta1)

        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        return m, v

    def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
        return group['lr']

    def adam_update(self, state: Dict[str, any], group: Dict[str, any], param: torch.nn.Parameter,
                    m: torch.Tensor, v: torch.Tensor):
        beta1, beta2 = group['betas']
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        lr = self.get_lr(state, group)

        if self.optimized_update:
            denominator = v.sqrt().add_(group['eps'])
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            param.data.addcdiv_(m, denominator, value=-step_size)
        else:
            denominator = (v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            step_size = lr / bias_correction1
            param.data.addcdiv_(m, denominator, value=-step_size)

    def step_param(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.nn.Parameter):
        grad = self.weight_decay(param, grad, group)
        m, v = self.get_mv(state, group, grad)
        state['step'] += 1
        self.adam_update(state, group, param, m, v)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamOptimizer does not support sparse gradients')

                state = self.state[param]

                if len(state) == 0:
                    self.init_state(state, group, param)

                self.step_param(state, group, grad, param)

        return loss