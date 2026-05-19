import builtins
import os

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def dprint(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master: bool):
    """
    Overrides the built-in print function to silence output on non-master ranks.

    Corner Case: Use `print(..., force=True)` to bypass this and force a print on all ranks.
    """
    builtin_print = builtins.print

    def custom_print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = custom_print


def ddp_all_skip(skip_local: bool, device: torch.device) -> bool:
    """
    Synchronizes a skip flag across all ranks using a MAX reduction.

    Corner Case: Extremely useful for preventing DDP hangs when one rank encounters
    a bad batch or OOM and needs to skip, while other ranks proceed normally.

    Returns:
        bool: True if ANY rank passed skip_local=True, False otherwise.
    """
    t = torch.tensor(int(skip_local), device=device, dtype=torch.int32)
    if is_dist_avail_and_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduces a tensor across all ranks (SUM by default).

    Corner Case: Clones the input tensor internally to avoid unintended in-place modifications.

    Returns:
        torch.Tensor: The reduced tensor, or the original tensor unmodified if DDP is inactive.
    """
    if not is_dist_avail_and_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= get_world_size()
    return rt


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduces a dictionary of tensors across all ranks.

    Corner Case: Keys are alphabetically sorted internally prior to stacking to guarantee
    a deterministic reduction order across all distributed ranks.

    Returns:
        dict: A new dictionary with reduced values, or the original dict if DDP is inactive.
    """
    if not is_dist_avail_and_initialized():
        return input_dict

    with torch.no_grad():
        names = sorted(input_dict.keys())
        values = torch.stack([input_dict[k] for k in names])
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        if average:
            values /= get_world_size()
        return {k: v for k, v in zip(names, values)}


def init_ddp() -> torch.device:
    """
    Initializes the PyTorch distributed process group based on standard environment variables.

    Automatically handles setup for both `torchrun` (RANK, WORLD_SIZE) and
    Slurm (SLURM_PROCID, SLURM_NTASKS) environments. Overrides `print` on non-master ranks.

    Returns:
        torch.device: The designated CUDA device for the current process, or CPU if CUDA is missing.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return torch.device("cpu")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = rank % torch.cuda.device_count()
    else:
        print("Running in single-GPU mode.")
        return torch.device("cuda:0")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    dist.barrier(device_ids=[local_rank])
    setup_for_distributed(rank == 0)

    return torch.device(f"cuda:{local_rank}")
