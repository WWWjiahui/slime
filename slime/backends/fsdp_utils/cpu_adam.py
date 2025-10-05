"""
CPU Adam utilities for FSDP training in slime.
"""

import torch


def create_cpu_adam_optimizer(model: torch.nn.Module, **optimizer_kwargs):
    """Create CPU Adam optimizer for FSDP model."""
    # Try to import CPU Adam from DeepSpeed or other libraries
    try:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        return DeepSpeedCPUAdam(model.parameters(), **optimizer_kwargs)
    except ImportError:
        try:
            from colossalai.optim import CPUAdam
            return CPUAdam(model.parameters(), **optimizer_kwargs)
        except ImportError:
            # Fallback to standard AdamW with manual CPU offload
            print("Warning: CPU Adam not available, using standard AdamW with manual CPU offload")
            return torch.optim.AdamW(model.parameters(), **optimizer_kwargs)


def offload_optimizer_to_cpu(optimizer: torch.optim.Optimizer):
    """Manually offload optimizer states to CPU."""
    if not optimizer.state:
        return
    
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    state[key] = value.cpu()


def load_optimizer_to_gpu(optimizer: torch.optim.Optimizer, device=None):
    """Manually load optimizer states to GPU."""
    if device is None:
        device = torch.cuda.current_device()
    
    if not optimizer.state:
        return
    
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
                    state[key] = value.to(device)
