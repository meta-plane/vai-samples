import torch
from torchvision.models import googlenet, GoogLeNet_Weights
from safetensors.torch import save_file


BN_EPS = 1e-3  # torchvision googlenet BatchNorm eps


def fold_conv_bn(sd, conv_prefix, bn_prefix):
    """Fold BatchNorm into Conv weights/bias."""
    w = sd[f"{conv_prefix}.weight"].detach().cpu()

    gamma = sd[f"{bn_prefix}.weight"].detach().cpu()
    beta = sd[f"{bn_prefix}.bias"].detach().cpu()
    mean = sd[f"{bn_prefix}.running_mean"].detach().cpu()
    var = sd[f"{bn_prefix}.running_var"].detach().cpu()

    inv_std = gamma / torch.sqrt(var + BN_EPS)
    w_fused = w * inv_std[:, None, None, None]
    b_fused = beta - mean * inv_std
    return w_fused, b_fused


def add_conv_bn(sd, conv_prefix, bn_prefix, dst_key, out):
    w, b = fold_conv_bn(sd, conv_prefix, bn_prefix)
    out[f"{dst_key}.weight"] = w
    out[f"{dst_key}.bias"] = b


def export(path="113-GoogleNet-Huicheol/weights.safetensors"):
    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    sd = model.state_dict()
    out = {}

    # Stem
    add_conv_bn(sd, "conv1.conv", "conv1.bn", "conv1", out)
    add_conv_bn(sd, "conv2.conv", "conv2.bn", "conv2_reduce", out)   # 1x1 reduce
    add_conv_bn(sd, "conv3.conv", "conv3.bn", "conv2", out)          # 3x3 conv

    # FC (no BN)
    out["fc.weight"] = sd["fc.weight"].detach().cpu()
    out["fc.bias"] = sd["fc.bias"].detach().cpu()

    def add_block(name, prefix):
        # 1x1
        add_conv_bn(sd, f"{name}.branch1.conv", f"{name}.branch1.bn", f"{prefix}.1x1", out)

        # 3x3 path: reduce + conv
        add_conv_bn(sd, f"{name}.branch2.0.conv", f"{name}.branch2.0.bn", f"{prefix}.3x3_reduce", out)
        add_conv_bn(sd, f"{name}.branch2.1.conv", f"{name}.branch2.1.bn", f"{prefix}.3x3", out)

        # 5x5 path: reduce + 5x5
        add_conv_bn(sd, f"{name}.branch3.0.conv", f"{name}.branch3.0.bn", f"{prefix}.5x5_reduce", out)
        add_conv_bn(sd, f"{name}.branch3.1.conv", f"{name}.branch3.1.bn", f"{prefix}.5x5", out)

        # pool proj (maxpool + 1x1 conv)
        add_conv_bn(sd, f"{name}.branch4.1.conv", f"{name}.branch4.1.bn", f"{prefix}.pool_proj", out)

    add_block("inception3a", "inception3a")
    add_block("inception3b", "inception3b")
    add_block("inception4a", "inception4a")
    add_block("inception4b", "inception4b")
    add_block("inception4c", "inception4c")
    add_block("inception4d", "inception4d")
    add_block("inception4e", "inception4e")
    add_block("inception5a", "inception5a")
    add_block("inception5b", "inception5b")

    save_file(out, path)


if __name__ == "__main__":
    export()
