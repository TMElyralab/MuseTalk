# Data preprocessing

Create two config yaml files, one for training and other for testing (both in same format as configs/inference/test.yaml)
The train yaml file should contain the training video paths and corresponding audio paths
The test yaml file should contain the validation video paths and corresponding audio paths

Run:
```
./data_new.sh train output train_video1.mp4 train_video2.mp4
./data_new.sh test output test_video1.mp4 test_video2.mp4
```
This creates folders which contain the image frames and npy files. This also creates train.json and val.json which can be used during the training.

## Data organization
```
./data/
├── images
│     └──RD_Radio10_000
│         └── 0.png
│         └── 1.png
│         └── xxx.png
│     └──RD_Radio11_000
│         └── 0.png
│         └── 1.png
│         └── xxx.png
├── audios
│     └──RD_Radio10_000
│         └── 0.npy
│         └── 1.npy
│         └── xxx.npy
│     └──RD_Radio11_000
│         └── 0.npy
│         └── 1.npy
│         └── xxx.npy
```

## Training
Simply run after preparing the preprocessed data
```
cd train_codes
sh train.sh #--train_json="../train.json" \(Generated in Data preprocessing step.)
            #--val_json="../val.json" \
```
## Inference with trained checkpoit
Simply run after training the model, the model checkpoints are saved at train_codes/output usually
```
python -m scripts.finetuned_inference --inference_config configs/inference/test.yaml --unet_checkpoint path_to_trained_checkpoint_folder
```

## TODO
- [x] release data preprocessing codes
- [ ] release some novel designs in training (after technical report)