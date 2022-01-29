# Very Little Memory Model
OBJECTIVE: TO ACHIEVE HIGHLY ACCURATE MODELS WITH VERY LITTLE MEMORY

BASE MODEL:RESNET

 - [x] Reversible network like Reformer
 - [ ] Cross-layer sharing like Albert

## Result on cifar100

|model|reversible|cross-layer sharing|Acc|Mem[MB]|
|---|---|---|---|---|
|resnet_18| | | | |
|resnet_18|:heavy_check_mark:| | | |
|resnet_18| |:heavy_check_mark:| | |
|resnet_18|:heavy_check_mark:|:heavy_check_mark:| | |
