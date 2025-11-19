from PIL import Image

def resize_and_save(input_path, output_path, size=(224, 224)):
    img = Image.open(input_path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    img.save(output_path)
    print(f"Saved resized image to {output_path}")

resize_and_save("zebra.jpg", "zebra_339.jpg")
