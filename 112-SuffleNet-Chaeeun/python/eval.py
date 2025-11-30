import os
import glob
import argparse
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import PIL
from PIL import Image

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

from network import ShuffleNetV2_Plus


def get_dump_dir() -> str:
    here = os.path.dirname(__file__)
    dump_dir = os.path.normpath(os.path.join(here, '..', 'dump'))
    os.makedirs(dump_dir, exist_ok=True)
    return dump_dir


class OpencvResize(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img)  # (H,W,3) RGB
        img = img[:, :, ::-1]  # to BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (
            int(self.size / H * W + 0.5),
            self.size,
        ) if H < W else (
            self.size,
            int(self.size / W * H + 0.5),
        )
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1]  # back to RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img


class ToBGRTensor(object):
    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:, :, ::-1]  # to BGR
        img = np.transpose(img, [2, 0, 1])  # (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img


def load_checkpoint(net, checkpoint_path, map_location=None):
    from collections import OrderedDict

    if checkpoint_path.lower().endswith('.safetensors'):
        if load_safetensors is None:
            raise RuntimeError('safetensors package is required to load .safetensors checkpoints')
        state = load_safetensors(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state = checkpoint.get('state_dict', checkpoint)
    expects_module = any(k.startswith('module.') for k in net.state_dict().keys())

    temp = OrderedDict()
    for k, v in state.items():
        has_module = k.startswith('module.')
        if expects_module and not has_module:
            temp['module.' + k] = v
        elif (not expects_module) and has_module:
            temp[k[len('module.'):]] = v
        else:
            temp[k] = v

    net.load_state_dict(temp, strict=True)


def print_model_summary(model: nn.Module, device: torch.device, input_size=(1, 3, 224, 224)):
    # Print a concise, readable per-layer summary using forward hooks (no extra deps)
    lines = []
    hooks = []

    def register_hook(module):
        # Only leaf modules
        if len(list(module.children())) > 0:
            return

        def hook(m, inp, out):
            name = m.__class__.__name__
            # Output shape(s)
            def shape_str(o):
                try:
                    return str(list(o.size()))
                except Exception:
                    return str(type(o))

            if isinstance(out, (list, tuple)):
                out_shape = '[' + ', '.join(shape_str(o) for o in out) + ']'
            else:
                out_shape = shape_str(out)

            # Parameter count for this module
            params = sum(p.numel() for p in m.parameters())
            lines.append((name, out_shape, params))

        hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)
    dummy = torch.zeros(*input_size, device=device)
    model.eval()
    with torch.no_grad():
        _ = model(dummy)
    for h in hooks:
        h.remove()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Render table
    print('==== Model Summary ====')
    print(f"{'Layer':30} {'Output Shape':25} {'Param #':>10}")
    print('-' * 70)
    for name, out_shape, params in lines:
        print(f"{name:30} {out_shape:25} {params:10d}")
    print('-' * 70)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")


def get_args():
    parser = argparse.ArgumentParser('ShuffleNetV2_Plus Eval')
    parser.add_argument('--input-dir', type=str, default=None, help='directory containing images to classify')
    parser.add_argument('--weights', type=str, default=None, help='path to weights (.pth/.pth.tar)')
    parser.add_argument('--model-size', type=str, default='Large', choices=['Small', 'Medium', 'Large'])
    parser.add_argument('--dump-debug', action='store_true', help='print intermediate activations for debugging')
    parser.add_argument('--dump-weights', action='store_true', help='print stem weights/bn stats for debugging')
    return parser.parse_args()


def print_debug(name: str, tensor: torch.Tensor, *, as_hwc: bool = False, sample_idx=None):
    t = tensor.detach().cpu()
    if as_hwc and t.dim() >= 3:
        if t.dim() == 4:
            t = t.squeeze(0)
        if t.dim() == 3:
            t = t.permute(1, 2, 0).contiguous()
    sampled = None
    if sample_idx is not None:
        try:
            sampled = t[tuple(sample_idx)].item()
        except Exception:
            sampled = None
    arr = t.reshape(-1)
    show = min(arr.numel(), 5)
    arr_list = arr[:show].tolist()
    shape = list(t.shape)
    msg = f'[debug][{name}] shape={shape} first={arr_list}'
    if sampled is not None:
        msg += f' sample{tuple(sample_idx)}={sampled}'
    print(msg)


def tensor_to_hwc_numpy(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu()
    if arr.dim() == 4:
        arr = arr.squeeze(0)
    if arr.dim() == 3:
        arr = arr.permute(1, 2, 0).contiguous()
    return arr.numpy()


def main():
    args = get_args()
    dump_dir = None
    save_outputs = args.dump_debug or args.dump_weights

    # Resolve defaults
    here = os.path.dirname(__file__)
    if args.input_dir is None:
        args.input_dir = os.path.normpath(os.path.join(here, '..', 'data'))

    resume = args.weights
    if (not resume) or (not os.path.exists(resume)):
        default_w = os.path.join(here, 'ShuffleNetV2+.Small.pth.tar')
        if os.path.exists(default_w):
            resume = default_w
            args.model_size = 'Small'

    # If a weights file is specified, infer model size from filename when using default
    if resume and os.path.exists(resume):
        name = os.path.basename(resume).lower()
        # Only auto-adjust if still at default Large
        if args.model_size == 'Large':
            if 'small' in name:
                args.model_size = 'Small'
            elif 'medium' in name:
                args.model_size = 'Medium'
            elif 'large' in name:
                args.model_size = 'Large'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    model = ShuffleNetV2_Plus(architecture=architecture, model_size=args.model_size)
    model = model.to(device)

    # Print model structure
    print('==== Model Architecture ====')
    print(model)
    # Also print a readable summary with per-layer outputs and params
    #print_model_summary(model, device, input_size=(1, 3, 224, 224))

    if resume and os.path.exists(resume):
        load_checkpoint(model, resume, map_location=None if torch.cuda.is_available() else 'cpu')
        if save_outputs:
            if dump_dir is None:
                dump_dir = get_dump_dir()
            def _print_weights(label, tensor):
                arr = tensor.detach().cpu()
                shape = list(arr.shape)
                flat = arr.reshape(-1)
                show = min(flat.numel(), 5)
                vals = flat[:show].tolist()
                print(f'[weights][{label}] shape={shape} first(flat[:{show}])={vals}')
            stem_conv = model.first_conv[0]
            stem_bn = model.first_conv[1]
            _print_weights('stem.conv.weight', stem_conv.weight)
            _print_weights('stem.bn.gamma', stem_bn.weight)
            _print_weights('stem.bn.beta', stem_bn.bias)
            np.save(os.path.join(dump_dir, 'stem_conv_weight_py.npy'), stem_conv.weight.detach().cpu().numpy())
            np.save(os.path.join(dump_dir, 'stem_bn_gamma_py.npy'), stem_bn.weight.detach().cpu().numpy())
            np.save(os.path.join(dump_dir, 'stem_bn_beta_py.npy'), stem_bn.bias.detach().cpu().numpy())
    # else: use randomly initialized weights

    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for p in patterns:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, p)))

    if not image_paths:
        print(f'No images found in {args.input_dir}')
        return

    tform = transforms.Compose([
        ToBGRTensor(),
    ])

    model.eval()
    with torch.no_grad():
        for img_path in sorted(image_paths):
            img = Image.open(img_path).convert('RGB')
            print(f"[PyTorch] source image size: {img.width}x{img.height}")
            x = tform(img).unsqueeze(0).to(device)
            if save_outputs and dump_dir is None:
                dump_dir = get_dump_dir()
            base = os.path.splitext(os.path.basename(img_path))[0] if save_outputs else None
            if save_outputs:
                np.save(os.path.join(dump_dir, f'{base}_input_py.npy'), tensor_to_hwc_numpy(x))
            if args.dump_debug:
                print_debug('input', x, as_hwc=True, sample_idx=(0, 0, 1))

            start = time.perf_counter()
            if args.dump_debug:
                stem_conv = model.first_conv[0](x)
                stem_bn = model.first_conv[1](stem_conv)
                out1 = model.first_conv[2](stem_bn)
                full_stem = out1
            else:
                out1 = model.first_conv(x)
                full_stem = out1

            feature_outs = []
            if args.dump_debug:
                tmp = out1
                for i, layer in enumerate(model.features):
                    tmp = layer(tmp)
                    feature_outs.append(tmp)
                out2 = tmp
            else:
                out2 = model.features(out1)

            if args.dump_debug:
                conv_last_conv = model.conv_last[0](out2)
                conv_last_bn = model.conv_last[1](conv_last_conv)
                out3 = model.conv_last[2](conv_last_bn)  # HS
            else:
                conv_last_conv = conv_last_bn = None
                out3 = model.conv_last(out2)
                conv_last_bn = model.conv_last[1](model.conv_last[0](out2))  # for saving consistency
            out4 = model.globalpool(out3)
            out5 = model.LastSE(out4)
            flat = out5.contiguous().view(out5.size(0), -1)
            out6 = model.fc(flat)
            out6 = model.dropout(out6)
            logits = model.classifier(out6)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            prob = torch.softmax(logits, dim=1)
            best_prob, pred = torch.max(prob, dim=1)
            pred = int(pred.item())
            best_prob = float(best_prob.item())

            print(f'[(PyTorch) ShuffleNet evaluation: 1 iterations] => {elapsed_ms:.0f}ms')
            if args.dump_debug:
                print_debug('stem.conv', stem_conv, as_hwc=True, sample_idx=(0, 0, 1))
                print_debug('stem.bn', stem_bn, as_hwc=True, sample_idx=(0, 0, 1))
                print_debug('first_conv', full_stem, as_hwc=True, sample_idx=(0, 0, 1))
                print_debug('first_hs', out1, as_hwc=True, sample_idx=(0, 0, 1))
                for idx, fo in enumerate(feature_outs):
                    print_debug(f'feat.{idx}', fo, as_hwc=True, sample_idx=(0, 0, 1))
                print_debug('conv_last', conv_last_conv if conv_last_conv is not None else out3, as_hwc=True, sample_idx=(0, 0, 1))
                print_debug('bn_last', conv_last_bn, as_hwc=True, sample_idx=(0, 0, 1))
                print_debug('last_hs', out3, as_hwc=True)
                print_debug('gap', out4, as_hwc=True)
                print_debug('lastSE', out5.squeeze(0))
                print_debug('logits', logits)
                np.save(os.path.join(dump_dir, f'{base}_stem_conv_py.npy'), tensor_to_hwc_numpy(stem_conv))
                np.save(os.path.join(dump_dir, f'{base}_stem_bn_py.npy'), tensor_to_hwc_numpy(stem_bn))
                np.save(os.path.join(dump_dir, f'{base}_first_conv_py.npy'), tensor_to_hwc_numpy(full_stem))
                for idx, fo in enumerate(feature_outs):
                    np.save(os.path.join(dump_dir, f'{base}_feat{idx}_py.npy'), tensor_to_hwc_numpy(fo))
                np.save(os.path.join(dump_dir, f'{base}_conv_last_py.npy'), tensor_to_hwc_numpy(conv_last_conv if conv_last_conv is not None else out3))
                np.save(os.path.join(dump_dir, f'{base}_bn_last_py.npy'), tensor_to_hwc_numpy(conv_last_bn))
            print(f'{os.path.basename(img_path)} -> {pred} (prob={best_prob:.4f})')


if __name__ == '__main__':
    main()
