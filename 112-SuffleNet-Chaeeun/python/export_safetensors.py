import os
import argparse
import torch
import numpy as np
from safetensors.torch import save_file


def _to_cpu_np(t):
    return t.detach().cpu().numpy()


def conv_to_vkB(w_tensor):
    """
    PyTorch Conv2d weights: [out, in, kH, kW]
    C++ ConvolutionNode expects B: [in*k*k, out]
    Returns np.float32 array of shape [I*K*K, O].
    """
    w = _to_cpu_np(w_tensor)
    O, I, kH, kW = w.shape
    # astype creates a new contiguous array
    return w.reshape(O, I * kH * kW).transpose(1, 0).astype(np.float32, copy=True)


def conv_to_vkB_dw(w_tensor):
    """
    Depthwise Conv2d weights: [C, 1, kH, kW]
    Expand to block-diagonal B with shape [C*K*K, C].
    """
    w = _to_cpu_np(w_tensor)
    C, I, kH, kW = w.shape
    assert I == 1, "Depthwise conv expected with groups=C (in per-filter=1)"
    K = kH * kW
    B = np.zeros((C * K, C), dtype=np.float32)
    kernels = w.reshape(C, K)
    for c in range(C):
        B[c * K:(c + 1) * K, c] = kernels[c]
    return B  # already contiguous


def linear_to_vkB(w_tensor):
    """
    PyTorch Linear weights: [out, in]
    C++ FC expects [in, out]
    """
    w = _to_cpu_np(w_tensor)
    O, I = w.shape
    return w.transpose(1, 0).astype(np.float32, copy=True)


def zeros_like_conv_out(w_tensor):
    O = int(_to_cpu_np(w_tensor).shape[0])
    return np.zeros((O,), dtype=np.float32)


def bn_to_dict(state, base):
    return {
        "gamma": _to_cpu_np(state[base + ".weight"]).astype(np.float32),
        "beta": _to_cpu_np(state[base + ".bias"]).astype(np.float32),
        "running_mean": _to_cpu_np(state[base + ".running_mean"]).astype(np.float32),
        "running_var": _to_cpu_np(state[base + ".running_var"]).astype(np.float32),
    }


def coerce_state_dict(checkpoint):
    """Return a flat state_dict with any leading 'module.' removed.
    Accepts a checkpoint dict or a state_dict directly.
    """
    state = checkpoint.get("state_dict", checkpoint)
    return { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }


def find_key_with_suffix(sd, suffix):
    for k in sd.keys():
        if k.endswith(suffix):
            return k
    return None


def _tn(x: np.ndarray) -> torch.Tensor:
    """Create a contiguous float32 torch tensor from numpy."""
    return torch.from_numpy(np.ascontiguousarray(x.astype(np.float32, copy=False))).contiguous()


def export_safetensors(ckpt_path, out_path, fmt="cpp"):
    sd_raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = coerce_state_dict(sd_raw)

    if fmt == "python":
        tensors = {k: _tn(_to_cpu_np(v)) for k, v in sd.items()}
        save_file(tensors, out_path)
        print(f"Exported SafeTensors (python keys): {out_path} (tensors={len(tensors)})")
        return

    T = {}

    # Stem (first_conv)
    stem_conv_key = (find_key_with_suffix(sd, "first_conv.0.weight") or
                     find_key_with_suffix(sd, "stem.0.weight"))
    if not stem_conv_key:
        raise KeyError("Could not locate stem conv weight (e.g., 'first_conv.0.weight') in checkpoint")
    stem_pref = stem_conv_key[: stem_conv_key.rfind("0.weight")]
    w = conv_to_vkB(sd[stem_conv_key])
    T["stem.conv.weight"] = _tn(w)
    T["stem.conv.bias"] = _tn(zeros_like_conv_out(sd[stem_conv_key]))
    for k, v in bn_to_dict(sd, stem_pref + "1").items():
        T[f"stem.bn.{k}"] = _tn(v)

    # Features blocks: detect all indices
    import re
    idxs = sorted({int(m.group(1)) for k in sd.keys() if (m:=re.match(r".*features\.(\d+)\.", k))})
    for i in idxs:
        bm = f"features.{i}.branch_main"
        bp = f"features.{i}.branch_proj"
        is_xc = (bm + ".12.weight") in sd  # Xception
        has_proj = (bp + ".0.weight") in sd

        if not is_xc:
            # ShuffleUnit
            # pw1/bn1
            pw1_w = sd[bm + ".0.weight"]
            T[f"features.{i}.pw1.weight"] = _tn(conv_to_vkB(pw1_w))
            T[f"features.{i}.pw1.bias"] = _tn(zeros_like_conv_out(pw1_w))
            for k, v in bn_to_dict(sd, bm + ".1").items():
                T[f"features.{i}.bn1.{k}"] = _tn(v)
            # dw/bn2 (depthwise)
            dw_w = sd[bm + ".3.weight"]
            T[f"features.{i}.dw.weight"] = _tn(conv_to_vkB_dw(dw_w))
            T[f"features.{i}.dw.bias"] = _tn(zeros_like_conv_out(dw_w))
            for k, v in bn_to_dict(sd, bm + ".4").items():
                T[f"features.{i}.bn2.{k}"] = _tn(v)
            # pw2/bn3
            pw2_w = sd[bm + ".5.weight"]
            T[f"features.{i}.pw2.weight"] = _tn(conv_to_vkB(pw2_w))
            T[f"features.{i}.pw2.bias"] = _tn(zeros_like_conv_out(pw2_w))
            for k, v in bn_to_dict(sd, bm + ".6").items():
                T[f"features.{i}.bn3.{k}"] = _tn(v)

            if has_proj:
                proj_dw_w = sd[bp + ".0.weight"]
                T[f"features.{i}.proj.dw.weight"] = _tn(conv_to_vkB_dw(proj_dw_w))
                T[f"features.{i}.proj.dw.bias"] = _tn(zeros_like_conv_out(proj_dw_w))
                for k, v in bn_to_dict(sd, bp + ".1").items():
                    T[f"features.{i}.proj.bn1.{k}"] = _tn(v)
                proj_pw_w = sd[bp + ".2.weight"]
                T[f"features.{i}.proj.pw.weight"] = _tn(conv_to_vkB(proj_pw_w))
                T[f"features.{i}.proj.pw.bias"] = _tn(zeros_like_conv_out(proj_pw_w))
                for k, v in bn_to_dict(sd, bp + ".3").items():
                    T[f"features.{i}.proj.bn2.{k}"] = _tn(v)

            # Optional SE in block
            # Find any key like features.i.branch_main.<N>.SE_opr.1.weight
            se_base = None
            for k in sd.keys():
                if k.startswith(bm + ".") and ".SE_opr.1.weight" in k:
                    se_base = k.split(".SE_opr")[0] + ".SE_opr"
                    break
            if se_base is not None:
                se1_w = sd[se_base + ".1.weight"]
                T[f"features.{i}.se.conv1.weight"] = _tn(conv_to_vkB(se1_w))
                T[f"features.{i}.se.conv1.bias"] = _tn(zeros_like_conv_out(se1_w))
                for k, v in bn_to_dict(sd, se_base + ".2").items():
                    T[f"features.{i}.se.bn.{k}"] = _tn(v)
                se2_w = sd[se_base + ".4.weight"]
                T[f"features.{i}.se.conv2.weight"] = _tn(conv_to_vkB(se2_w))
                T[f"features.{i}.se.conv2.bias"] = _tn(zeros_like_conv_out(se2_w))

        else:
            # ShuffleXception
            dw1_w = sd[bm + ".0.weight"]
            T[f"features.{i}.dw1.weight"] = _tn(conv_to_vkB_dw(dw1_w))
            T[f"features.{i}.dw1.bias"] = _tn(zeros_like_conv_out(dw1_w))
            for k, v in bn_to_dict(sd, bm + ".1").items():
                T[f"features.{i}.bn1.{k}"] = _tn(v)
            pw1_w = sd[bm + ".2.weight"]
            T[f"features.{i}.pw1.weight"] = _tn(conv_to_vkB(pw1_w))
            T[f"features.{i}.pw1.bias"] = _tn(zeros_like_conv_out(pw1_w))
            for k, v in bn_to_dict(sd, bm + ".3").items():
                T[f"features.{i}.bn1p.{k}"] = _tn(v)
            dw2_w = sd[bm + ".5.weight"]
            T[f"features.{i}.dw2.weight"] = _tn(conv_to_vkB_dw(dw2_w))
            T[f"features.{i}.dw2.bias"] = _tn(zeros_like_conv_out(dw2_w))
            for k, v in bn_to_dict(sd, bm + ".6").items():
                T[f"features.{i}.bn2.{k}"] = _tn(v)
            pw2_w = sd[bm + ".7.weight"]
            T[f"features.{i}.pw2.weight"] = _tn(conv_to_vkB(pw2_w))
            T[f"features.{i}.pw2.bias"] = _tn(zeros_like_conv_out(pw2_w))
            for k, v in bn_to_dict(sd, bm + ".8").items():
                T[f"features.{i}.bn2p.{k}"] = _tn(v)
            dw3_w = sd[bm + ".10.weight"]
            T[f"features.{i}.dw3.weight"] = _tn(conv_to_vkB_dw(dw3_w))
            T[f"features.{i}.dw3.bias"] = _tn(zeros_like_conv_out(dw3_w))
            for k, v in bn_to_dict(sd, bm + ".11").items():
                T[f"features.{i}.bn3.{k}"] = _tn(v)
            pw3_w = sd[bm + ".12.weight"]
            T[f"features.{i}.pw3.weight"] = _tn(conv_to_vkB(pw3_w))
            T[f"features.{i}.pw3.bias"] = _tn(zeros_like_conv_out(pw3_w))
            for k, v in bn_to_dict(sd, bm + ".13").items():
                T[f"features.{i}.bn3p.{k}"] = _tn(v)

            if has_proj:
                proj_dw_w = sd[bp + ".0.weight"]
                T[f"features.{i}.proj.dw.weight"] = _tn(conv_to_vkB_dw(proj_dw_w))
                T[f"features.{i}.proj.dw.bias"] = _tn(zeros_like_conv_out(proj_dw_w))
                for k, v in bn_to_dict(sd, bp + ".1").items():
                    T[f"features.{i}.proj.bn1.{k}"] = _tn(v)
                proj_pw_w = sd[bp + ".2.weight"]
                T[f"features.{i}.proj.pw.weight"] = _tn(conv_to_vkB(proj_pw_w))
                T[f"features.{i}.proj.pw.bias"] = _tn(zeros_like_conv_out(proj_pw_w))
                for k, v in bn_to_dict(sd, bp + ".3").items():
                    T[f"features.{i}.proj.bn2.{k}"] = _tn(v)

            # Optional SE in block
            se_base = None
            for k in sd.keys():
                if k.startswith(bm + ".") and ".SE_opr.1.weight" in k:
                    se_base = k.split(".SE_opr")[0] + ".SE_opr"
                    break
            if se_base is not None:
                se1_w = sd[se_base + ".1.weight"]
                T[f"features.{i}.se.conv1.weight"] = _tn(conv_to_vkB(se1_w))
                T[f"features.{i}.se.conv1.bias"] = _tn(zeros_like_conv_out(se1_w))
                for k, v in bn_to_dict(sd, se_base + ".2").items():
                    T[f"features.{i}.se.bn.{k}"] = _tn(v)
                se2_w = sd[se_base + ".4.weight"]
                T[f"features.{i}.se.conv2.weight"] = _tn(conv_to_vkB(se2_w))
                T[f"features.{i}.se.conv2.bias"] = _tn(zeros_like_conv_out(se2_w))

    # conv_last
    last_conv_key = (find_key_with_suffix(sd, "conv_last.0.weight") or
                     find_key_with_suffix(sd, "conv_last.0.weight"))
    if not last_conv_key:
        raise KeyError("Could not locate conv_last.0.weight in checkpoint")
    last_pref = last_conv_key[: last_conv_key.rfind("0.weight")]
    last_conv_w = sd[last_conv_key]
    T["last.conv.weight"] = _tn(conv_to_vkB(last_conv_w))
    T["last.conv.bias"] = _tn(zeros_like_conv_out(last_conv_w))
    for k, v in bn_to_dict(sd, last_pref + "1").items():
        T[f"last.bn.{k}"] = _tn(v)

    # Last SE
    se1_key = find_key_with_suffix(sd, "LastSE.SE_opr.1.weight")
    if se1_key:
        se_pref = se1_key[: se1_key.rfind("LastSE.SE_opr.1.weight")]
        se1_w = sd[se1_key]
        T["lastSE.conv1.weight"] = _tn(conv_to_vkB(se1_w))
        T["lastSE.conv1.bias"] = _tn(zeros_like_conv_out(se1_w))
        for k, v in bn_to_dict(sd, se_pref + "LastSE.SE_opr.2").items():
            T[f"lastSE.bn.{k}"] = _tn(v)
        se2_w = sd[se_pref + "LastSE.SE_opr.4.weight"]
        T["lastSE.conv2.weight"] = _tn(conv_to_vkB(se2_w))
        T["lastSE.conv2.bias"] = _tn(zeros_like_conv_out(se2_w))

    # FCs
    fc_w_key = find_key_with_suffix(sd, "fc.0.weight")
    if not fc_w_key:
        raise KeyError("Could not locate fc.0.weight in checkpoint")
    fc_w = sd[fc_w_key]
    T["fc1.weight"] = _tn(linear_to_vkB(fc_w))
    out_dim = _to_cpu_np(fc_w).shape[0]
    T["fc1.bias"] = torch.zeros((out_dim,), dtype=torch.float32)

    clf_w_key = find_key_with_suffix(sd, "classifier.0.weight")
    if not clf_w_key:
        raise KeyError("Could not locate classifier.0.weight in checkpoint")
    clf_w = sd[clf_w_key]
    T["classifier.weight"] = _tn(linear_to_vkB(clf_w))
    out_dim_c = _to_cpu_np(clf_w).shape[0]
    T["classifier.bias"] = torch.zeros((out_dim_c,), dtype=torch.float32)

    # Save
    save_file(T, out_path)
    print(f"Exported SafeTensors: {out_path} (tensors={len(T)})")


def main():
    here = os.path.dirname(__file__)
    parser = argparse.ArgumentParser("Export ShuffleNetV2+ weights to SafeTensors")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(here, "ShuffleNetV2+.Small.pth.tar"))
    parser.add_argument("--out", type=str, default=os.path.join(here, "weights.safetensors"))
    parser.add_argument("--format", type=str, default="cpp", choices=["cpp", "python"], help="cpp=for Vulkan C++ runtime, python=original state_dict keys")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    export_safetensors(args.checkpoint, args.out, fmt=args.format)


if __name__ == "__main__":
    main()
