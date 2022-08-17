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
# ToDo
* Mobileone FPN Train Exp
* Mobileone No FPN Train Exp
* Full COCO MV3 FPN TRAIN using Determined.ai
* Full COCO Mobileone FPN TRAIN using Determined.ai