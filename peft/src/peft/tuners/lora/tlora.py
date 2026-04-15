
from typing import Callable, Iterable, Optional
import torch
import torch.nn as nn
from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.model import LoraModel

def target_modules(model: nn.Module, config: LoraConfig) -> Iterable[nn.Module]:
    for name, module in model.named_modules():
        if LoraModel._check_target_module_exists(config, name) and isinstance(module, nn.Linear):
            yield name, module


def preprocess_tlora(
    model: nn.Module,
    config: LoraConfig,
    run_model: Optional[Callable[[], None]],
):
    if run_model is None:
        raise ValueError("run_model must be specified when covariance file and cache file aren't built.")
    hook_model = model

    def hook(module, input, output):

        input = input[0].detach().squeeze(0).data ## (context_length = 2048, dim)
        input = (input - input.mean(dim=0, keepdim=True))
        if torch.isnan(input).any() or torch.isinf(input).any():
            raise ValueError("Invalid value found in input, please check your input data.")
        new_cov = input.t() @ input
        n = module.sample_count
        if n == 0:
            module.covariance_matrix = new_cov
        else:
            alpha = 1.0 / (n + 1)
            beta = n / (n + 1)
            module.covariance_matrix.mul_(beta).add_(alpha * new_cov)
        if torch.isnan(module.covariance_matrix).any() or torch.isinf(module.covariance_matrix).any():
            raise ValueError("Invalid value found in covariance_matrix.")
        module.sample_count += 1
        del  input

    def backward_hook(module, grad_input, grad_output):

        for _, p in module.named_parameters():
                new_grad = p.grad
                n = module.sample_count - 1
                weight = module.weight.data
                if p.grad.ndim < 2:
                    continue
                score = torch.abs(weight * new_grad).mean().item()
                if n == 0:
                    module.score = score
                else:
                    alpha = 1.0 / (n + 1)
                    beta = n / (n + 1)
                    module.score = module.score*beta + score*alpha
        del new_grad, weight


    handles = []
    for name, module in target_modules(hook_model, config):
        module.sample_count = 0
        module.covariance_matrix = 0
        module.score = 0
        handles.append(module.register_forward_hook(hook))
        handles.append(module.register_full_backward_hook(backward_hook))  # 注册反向钩子
    run_model()

    # Clear the hooks
    for handle in handles:
        handle.remove()


    sum_alpha = 0
    sum_r = 0
    sum_contribution = 0
    for name, module in target_modules(model, config):
        sum_alpha += config.lora_alpha
        sum_r += config.r
        sum_contribution+=module.score
    rank_list = [round(sum_r * module.score / sum_contribution) for name,module in target_modules(model, config)]

    missing_rank = sum_r - sum(rank_list)
    while missing_rank != 0:

        max_score_module = rank_list.index(max(rank_list))
        if missing_rank >0:
            rank_list[max_score_module] += 1
            missing_rank -= 1
        else:
            rank_list[max_score_module] -= 1
            missing_rank += 1

    for (name, module), rank in zip(target_modules(model, config), rank_list):
        module.rank = rank
        alpha = round(sum_alpha * module.score / sum_contribution)
        module.a = max(config.lora_alpha, alpha)
