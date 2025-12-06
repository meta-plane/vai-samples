import torch
from torchvision.models import googlenet, GoogLeNet_Weights
from safetensors.torch import save_file

def add_conv(sd, src_key, dst_key, out):
    w = sd[f"{src_key}.weight"].detach().cpu()
    out[f"{dst_key}.weight"] = w
    b_key = f"{src_key}.bias"
    out[f"{dst_key}.bias"] = sd[b_key].detach().cpu() if b_key in sd else torch.zeros(w.shape[0])

def export(path="113-GoogleNet-Huicheol/weights.safetensors"):
    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    sd = model.state_dict()
    out = {}
    add_conv(sd, "conv1.conv", "conv1", out)
    add_conv(sd, "conv2.conv", "conv2_reduce", out)   # 1x1 reduce
    add_conv(sd, "conv3.conv", "conv2", out)          # 3x3 conv
    add_conv(sd, "fc", "fc", out)

    def add_block(name, prefix):
        # 1x1
        add_conv(sd, f"{name}.branch1.conv",      f"{prefix}.1x1", out)

        # 3x3 path: reduce + conv
        add_conv(sd, f"{name}.branch2.0.conv",    f"{prefix}.3x3_reduce", out)
        add_conv(sd, f"{name}.branch2.1.conv",    f"{prefix}.3x3",        out)

        # 5x5 path: reduce + conv
        add_conv(sd, f"{name}.branch3.0.conv",    f"{prefix}.5x5_reduce", out)
        add_conv(sd, f"{name}.branch3.1.conv",    f"{prefix}.5x5",        out)

        # pool proj (maxpool 뒤 1x1 conv)
        add_conv(sd, f"{name}.branch4.1.conv",    f"{prefix}.pool_proj",  out)

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
