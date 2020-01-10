## MS-TCN
this is the re-implementation of Mult-Stage TCN (MS-TCN) with pytorch.

## Dataset
GTEA, 50Salads, Breakfast  

**Maximum number of frames in a video**
GTEA: 2009
50salads: 9072 (downsampled from 30fps to 15fps)
breakfast: 9741

You can download features and G.T. of these datasets from [this repository](https://github.com/yabufarha/ms-tcn)

## Requirements
```
* Python 3.x
* pytorch => 1.0
* torchvision
* pandas
* numpy
* tqdm
* PyYAML
* addict
```

You can download packages using requirements.txt.  
``` pip install -r requirements.txt```


## directory structure
```
basestation_rust_detection ── csv/
                           ├─ libs/
                           ├─ result/
                           ├─ utils/
                           ├─ dataset ─── 50salads/...
                           │           ├─ breakfast/...
                           │           └─ gtea ─── features/
                           │                    ├─ groundTruth/
                           │                    ├─ splits/
                           │                    └─ mapping.txt
                           ├.gitignore
                           ├ README.md
                           ├ requirements.txt
                           ├ train.py
                           └ eval.py
```

## How to use
### Setting
First, convert ground truth files into numpy array.
Please run `python utils/generate_gt_array.py ./dataset`
Then, please run the below script to generate csv files for data laoder'.
`python utils/builda_dataset.py ./dataset`

### Training
Just run `python train.py ./result/xxx/xxx/config.yaml --resume`

You can train a model in your own setting.
Follow the below example of a configuration file.

```model: ms-tcn
stages: ['dilated', 'dilated', 'dilated', 'dilated']
n_features: 64
dilated_n_layers: 10
kernel_size: 15

# loss function
ce: True    # cross entropy
tmse: True    # temporal mse
tmse_weight: 0.15

class_weight: True    # if you use class weight to calculate cross entropy or not

batch_size: 1

# the number of input feature channels
in_channel: 2048

# thresholds for calcualting F1 Score
thresholds: [0.1, 0.25, 0.5]

num_workers: 4
max_epoch: 50

optimizer: Adam
scheduler: None

learning_rate: 0.0005
lr_patience: 10       # Patience of LR scheduler
momentum: 0.9         # momentum of SGD
dampening: 0.0        # dampening for momentum of SGD
weight_decay: 0.0001  # weight decay
nesterov: True        # enables Nesterov momentum
final_lr: 0.1         # final learning rate for AdaBound
poly_power: 0.9       # for polunomial learning scheduler

param_search: False

dataset: 50salads
dataset_dir: ./dataset
csv_dir: ./csv
split: 1

result_path: ./result/50salads/ms-tcn/split1
```

### Test
Run `python eval.py ./result/xxx/xxx/config.yaml test`

### average cross validation results
Run `python utils/average_cv_results.py [result_dir]`


## References
* Colin Lea et al., "Temporal Convolutional Networks for Action Segmentation and Detection", in CVPR2017 [paper](http://zpascal.net/cvpr2017/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.pdf)
* Yazan Abu Farha et al., "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation", in CVPR2019 [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.pdf) [code](https://github.com/yabufarha/ms-tcn)