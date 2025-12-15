import numpy as np
import torch

def save_tensor_bin(path: str, tensor: torch.Tensor):
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # CPU / float32 강제
    tensor = tensor.detach().cpu().float()

    shape = tensor.shape
    rank = len(shape)

    with open(path, "wb") as f:
        # rank
        np.array([rank], dtype="<u4").tofile(f)

        # shape
        np.array(shape, dtype="<u4").tofile(f)

        # data
        tensor.numpy().astype("<f4").tofile(f)


def load_tensor_bin(path: str, device="cpu") -> torch.Tensor:
    with open(path, "rb") as f:
        # rank
        rank = int(np.fromfile(f, dtype="<u4", count=1)[0])
        if rank <= 0 or rank > 32:
            raise ValueError(f"Invalid rank: {rank}")

        # shape
        shape = np.fromfile(f, dtype="<u4", count=rank)
        shape = tuple(int(x) for x in shape)

        # elem count
        elem_count = 1
        for d in shape:
            elem_count *= d

        # data
        data = np.fromfile(f, dtype="<f4", count=elem_count)
        if data.size != elem_count:
            raise ValueError("File ended unexpectedly")

    tensor = torch.from_numpy(data).reshape(shape)
    return tensor.to(device)