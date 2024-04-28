# Draft training codes

We provde the draft training codes here. Unfortunately, data preprocessing code is still being reorganized.

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