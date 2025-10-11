"""
CPU Adam utilities for FSDP training in slime.
"""

import torch


def ensure_fsdp_params_on_cpu(model: torch.nn.Module):
    """Ensure FSDP model parameters are accessible on CPU for CPU Adam."""
    cpu_params = []
    for param in model.parameters():
        if hasattr(param, 'full_tensor'):
            # Handle DTensor case - get the full tensor and ensure it's a leaf tensor
            full_tensor = param.full_tensor()
            if full_tensor.is_cuda:
                # Create a new leaf tensor on CPU
                cpu_param = torch.empty_like(full_tensor, device='cpu', requires_grad=True)
                cpu_param.data.copy_(full_tensor.data)
                cpu_params.append(cpu_param)
            else:
                cpu_params.append(full_tensor)
        else:
            # Handle regular tensor case
            if param.is_cuda:
                # Create a new leaf tensor on CPU
                cpu_param = torch.empty_like(param, device='cpu', requires_grad=True)
                cpu_param.data.copy_(param.data)
                cpu_params.append(cpu_param)
            else:
                cpu_params.append(param)
    return cpu_params


def create_cpu_adam_optimizer(model: torch.nn.Module, **optimizer_kwargs):
    """Create CPU Adam optimizer for FSDP model."""
    # Try to import CPU Adam from DeepSpeed or other libraries
    # try:
    from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
    # DeepSpeed CPU Adam requires parameters to be on CPU
    # For FSDP models, we need to ensure parameters are accessible on CPU
    cpu_params = ensure_fsdp_params_on_cpu(model)
    optimizer = DeepSpeedCPUAdam(cpu_params, **optimizer_kwargs)
    print("Successfully created DeepSpeed CPU Adam optimizer")
    return optimizer


# Note: Manual offload/onload functions removed as torch_memory_saver handles memory management


def is_cpu_adam_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    """Check if the optimizer is a DeepSpeed CPU Adam optimizer."""
    return hasattr(optimizer, 'ds_opt_adam')
