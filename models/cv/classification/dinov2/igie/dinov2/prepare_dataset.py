from dinov2.data.datasets import ImageNet

root = "/data/deepspark_dataset/ILSVRC2012"
extra = "/data/deepspark_dataset/ILSVRC2012/extra"

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root=root, extra=extra)
    dataset.dump_extra()