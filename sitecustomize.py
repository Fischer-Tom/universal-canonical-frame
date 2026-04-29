import torch

_old_tensor_repr = torch.Tensor.__repr__


def _tensor_repr_with_shape(self, *, tensor_contents=None):
    try:
        shape = tuple(self.shape)
        dtype = self.dtype
        device = self.device
        grad = self.requires_grad
        base = _old_tensor_repr(self, tensor_contents=tensor_contents)

        # Keep value preview short, prepend the metadata you actually care about.
        return (
            f"Tensor(shape={shape}, dtype={dtype}, device={device}, "
            f"requires_grad={grad}) {base}"
        )
    except Exception:
        # Fall back to the original repr if anything goes sideways.
        return _old_tensor_repr(self, tensor_contents=tensor_contents)


torch.Tensor.__repr__ = _tensor_repr_with_shape
