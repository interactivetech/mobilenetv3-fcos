# scaling learn rate (0.01/8*2)*8
# for training 2 images per gpu, on an 8 gpu machine, with a combined lr of 0.01,
# the total batch size is 16 the per batch learn rate is: 0.01/8*2 = 0.000625
# scaling that to only a batch size of 8: 2 images per gpu, only 4 gpus
# learn rate is 0.005
# mobileone wont converge if learning rate is greater than 1e-4
torchrun --nproc_per_node=4 train.py \
    --dataset coco \
    --data-path=/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/ \
    --model mobileone_s4_fcos \
    --epochs 26 \
    --lr-steps 16 22 \
    --aspect-ratio-group-factor 3 \
    --lr 0.0001 \
    --batch-size 8 \
    --workers 2 \
    --amp \
    --weights-backbone ResNet50_Weights.IMAGENET1K_V1
