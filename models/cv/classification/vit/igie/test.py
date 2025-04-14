from transformers import ViTImageProcessor, ViTForImageClassification
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

print(model)
