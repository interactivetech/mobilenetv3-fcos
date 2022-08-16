torchrun --nproc_per_node=4 train.py \
    --dataset coco \
    --data-path=/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/ \
    --model mobilenet_v3_fcos \
    --epochs 26 \
    --lr-steps 16 22 \
    --aspect-ratio-group-factor 3 \
    --lr 0.01 \
    --amp \
    --weights-backbone ResNet50_Weights.IMAGENET1K_V1
