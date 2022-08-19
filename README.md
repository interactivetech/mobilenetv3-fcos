# mobilenetv3-fcos
Test using Torchvision API to train an FCOS model with MobilenetV3 FPN backbone

# Finetune on PennFudanPed Dataset
* `bash install_pennfudanped_dataset.sh`
* `python torchvision_finetuning_fcos.py`

# Train on Full COCO Dataset:
* `bash train.sh`

# Result Training for 26 Epochs
Dataset to train (20% of COCO Dataset) `instances_train2017_minicoco.json` from: https://github.com/giddyyupp/coco-minitrain
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.20658
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.36225
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.20875
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.10145
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.22442
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.27279
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.21949
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.38100
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.41197
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.21135
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.44376
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.57073
Training time 8:16:55
```

# Result Training MobilenetV3 (No FPN) on MiniCOCO Train
```
Test: Total time: 0:00:43 (0.0350 s / it)
Averaged stats: model_time: 0.0244 (0.0258)  evaluator_time: 0.0054 (0.0075)
Accumulating evaluation results...
DONE (t=5.87s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.14914
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.28572
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.13886
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.02752
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.15849
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.27897
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.15520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.23457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.24191
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.03601
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.23545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.48418
Training time 2:46:21
```

# MobileOne S4 No FPN Train (23 Epochs)
```
DONE (t=8.01s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.11387
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.25558
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.08531
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.05117
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.16744
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.16577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.13663
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.24625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.26201
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.06828
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.32610
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.42676
```

Train Settings:
```
torchrun --nproc_per_node=4 train.py \
    --dataset coco \
    --data-path=/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/ \
    --model mobileone_s4_fcos \
    --epochs 26 \
    --lr-steps 16 22 \
    --aspect-ratio-group-factor 3 \
    --lr 0.0005 \
    --batch-size 7 \
    --workers 2 \
    --amp \
    --weights-backbone ResNet50_Weights.IMAGENET1K_V1
```

# Mobileone FPN Train Exp
'''
Averaged stats: model_time: 0.2638 (0.2495)  evaluator_time: 0.0041 (0.0051)
Accumulating evaluation results...
DONE (t=3.77s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.15720
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.30767
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.14181
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.07788
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.19181
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.19617
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.16918
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.27215
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.27527
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.09949
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.30088
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.41984
Training time 1 day, 4:32:46
'''

Training Settings:
'''
torchrun --nproc_per_node=4 train.py \
    --dataset coco \
    --data-path=/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/ \
    --model mobileone_s4_fpn_fcos \
    --epochs 26 \
    --lr-steps 16 22 \
    --aspect-ratio-group-factor 3 \
    --lr 0.001 \
    --batch-size 1 \
    --workers 2 \
    --amp \
    --weights-backbone ResNet50_Weights.IMAGENET1K_V1
'''
# ToDo
* Full COCO MV3 FPN TRAIN using Determined.ai
* Full COCO Mobileone FPN TRAIN using Determined.ai