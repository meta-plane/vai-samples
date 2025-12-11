from safetensors import safe_open

path = "weight/unet.safetensors"

with safe_open(path, framework="pt", device="cpu") as f:
    # print("텐서 이름들: \n", f.keys())
    for name in f.keys():
        t = f.get_tensor(name)
        print(name)