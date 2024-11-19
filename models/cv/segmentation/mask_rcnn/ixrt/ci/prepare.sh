ln -s /root/data/checkpoints/maskrcnn.wts ./python/
ln -s /root/data/datasets/coco ./coco
if [ "$1" = "nvidia" ]; then
    cd scripts && bash init_nv.sh
else
    cd scripts && bash init.sh
fi