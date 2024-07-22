# Pedestrian-Attribute-Recognition-as-Label-balanced-Multi-label-Learning
This is the official pytorch implementation of the ICML2024 main conference paper: 'Pedestrian Attribute Recognition as Label-balanced Multi-label Learning' at https://arxiv.org/abs/2405.04858. By following steps, researchers can ensure a smooth reproduction of our experimental results.

For a rigorous and unbiased comparison, our study adheres strictly to the benchmarking guidelines outlined in https://github.com/valencebond/Rethinking_of_PAR, by adopting the publicly available code of this baseline work for dataset partitioning, data loader generation, backbone configuration and metric setup. In order to facilitate an easy implementation of our method, we recommend to review the public code repository at https://github.com/valencebond/Rethinking_of_PAR and implement our method directly onto this code by simply replacing serval .py files of training setting and configs:


• Replacing the original 'train.py' file with ours.

• Replacing the original 'batch_engine.py' file with ours.

• Replacing the original 'tools/function.py' file by ours.

• Replacing the original 'models/base_block.py' file with ours.

• Replacing the original 'configs/default.py' file with ours.

• Replacing the original 'configs/pedes_baseline' folder by ours.

• Putting the 'convnext.py' file under the path 'models/backbone/' for testing on ConvNeXt.

# Environment
Pytorch == 1.10.1+cu102, numpy == 1.19.5 and python == 3.6.9 64-bit. All experiments in main text are conducted on a single NVIDIA Tesla V100 32G. 

# Datasets
Please download the datasets (PA100k, RAP and PETA) from their official sources and structure them according to the specifications demanded in https://github.com/valencebond/Rethinking_of_PAR:

• PETA@http://mmlab.ie.cuhk.edu.hk/projects/PETA.html

• PA100k@https://github.com/xh-liu/HydraPlus-Net

• RAPv1@http://www.rapdataset.com

• UPAR@https://github.com/speckean/upar_challenge/tree/main

For PETAzs and RAPzs datasets specifically, this baseline work conveniently provides their re-organizing files under the 'data' directory. We also give the code to re-organize balanced testsets like that for ImageNet-LT and CIFAR100-LT in the folder `balanced_setting', you could just simply replace the corresponding codes in https://github.com/valencebond/Rethinking_of_PAR with those in this folder to get the results.

# Training and Testing
Please pre-train an arbitary baseline model at first (remove the FRDL and GOAT modules in our code to just train a baseline backbone), importantly, without any weighted BCE loss, and save the converged model into the 'model_path' variable defined under the 'get_reload_weight' function in the 'function.py' file. Please note that, you could use any pre-trained feature extractor of PAR in this step, not necessarily be the baseline model above for better results. Next, simply run the following command,
   
```
CUDA_VISIBLE_DEVICES=0 python train.py --cfg ./configs/pedes_baseline/DATASET_CONFIG
```

where DATASET_CONFIG can be any config file within the 'pedes_baseline' folder. Sequentially, all the training process of FRDL and GOAT would be automatically operated, along with corresponding benchmark results to be displayed. Config files named with '_base' is applied for the feature extractor training in the Stage#1 of FRDL.


