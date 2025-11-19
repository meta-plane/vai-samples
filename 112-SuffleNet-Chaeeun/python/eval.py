import os
import glob
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import PIL
from PIL import Image

from network import ShuffleNetV2_Plus


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


def load_checkpoint(net, checkpoint):
    from collections import OrderedDict

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
    return parser.parse_args()


def main():
    args = get_args()

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
        checkpoint = torch.load(resume, map_location=None if torch.cuda.is_available() else 'cpu')
        load_checkpoint(model, checkpoint)
    # else: use randomly initialized weights

    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for p in patterns:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, p)))

    if not image_paths:
        print(f'No images found in {args.input_dir}')
        return

    tform = transforms.Compose([
        OpencvResize(256),
        transforms.CenterCrop(224),
        ToBGRTensor(),
    ])

    model.eval()
    with torch.no_grad():
        for img_path in sorted(image_paths):
            img = Image.open(img_path).convert('RGB')
            x = tform(img).unsqueeze(0).to(device)
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())
            print(f'{os.path.basename(img_path)} -> {pred}')


if __name__ == '__main__':
    main()
