# Draft training codes

We provde the draft training codes here. Unfortunately, data preprocessing code is still being reorganized.

## Setup

We trained our model on an NVIDIA A100 with `batch size=8, gradient_accumulation_steps=4` for 20w+ steps. Using multiple GPUs should accelerate the training.

## Data preprocessing
 You could refer the inference codes which [crop the face images](https://github.com/TMElyralab/MuseTalk/blob/main/scripts/inference.py#L79) and [extract audio features](https://github.com/TMElyralab/MuseTalk/blob/main/scripts/inference.py#L69).

Finally, the data should be organized as follows:
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
sh train.sh
```

## TODO
- [ ] release data preprocessing codes
- [ ] release some novel designs in training (after technical report)