# Data preprocessing

Create two config yaml files, one for training and other for testing (both in same format as configs/inference/test.yaml)
The train yaml file should contain the training video paths and corresponding audio paths
The test yaml file should contain the validation video paths and corresponding audio paths

Run:
```
python -m scripts.data --inference_config path_to_train.yaml --folder_name train
python -m scripts.data --inference_config path_to_test.yaml --folder_name test
```
This creates folders which contain the image frames and npy files.


## Data organization
```
./data/
├── images
│     └──train
│         └── 0.png
│         └── 1.png
│         └── xxx.png
│     └──test
│         └── 0.png
│         └── 1.png
│         └── xxx.png
├── audios
│     └──train
│         └── 0.npy
│         └── 1.npy
│         └── xxx.npy
│     └──test
│         └── 0.npy
│         └── 1.npy
│         └── xxx.npy
```

## Training
Simply run after preparing the preprocessed data
```
sh train.sh
```
## Inference with trained checkpoit
Simply run after training the model, the model checkpoints are saved at train_codes/output usually
```
python -m scripts.finetuned_inference --inference_config configs/inference/test.yaml --unet_checkpoint path_to_trained_checkpoint_folder
```

## TODO
- [x] release data preprocessing codes
- [ ] release some novel designs in training (after technical report)