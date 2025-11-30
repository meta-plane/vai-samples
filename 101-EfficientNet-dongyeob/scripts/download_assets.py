import os
import urllib.request

def download_file(url, path):
    if os.path.exists(path):
        print(f"{path} already exists.")
        return
    print(f"Downloading {url} to {path}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
            out_file.write(response.read())
        print("Done.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # Giant Panda image
    image_url = "https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG"
    image_path = os.path.join(assets_dir, "panda.jpg")
    download_file(image_url, image_path)

    # ImageNet labels
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels_path = os.path.join(assets_dir, "imagenet_classes.txt")
    download_file(labels_url, labels_path)

if __name__ == "__main__":
    main()
