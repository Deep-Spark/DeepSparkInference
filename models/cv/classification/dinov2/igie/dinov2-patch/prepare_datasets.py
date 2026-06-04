from dinov2.data.datasets import ImageNet

root = "/mnt/deepspark/data/datasets/imagenet"
extra = "/mnt/deepspark/data/datasets/imagenet/extra"

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root=root, extra=extra)
    dataset.dump_extra()