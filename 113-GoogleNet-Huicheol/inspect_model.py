import torchvision.models as models

model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
print(model)
