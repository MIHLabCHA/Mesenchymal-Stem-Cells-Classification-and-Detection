# Mesenchymal Stem Cells Classification and Detection

## Summary
This repository contains the source of the our paper, currently under review of the [International Journal of Stem Cells](https://www.ijstemcell.com/main.html).

## Environment setting
### Windows
```
conda env create -n ENV_NAME --file=environment_windows.yml
conda activate ENV_NAME
```

### Linux
```
conda env create -n ENV_NAME --file=environment_linux.yml
conda activate ENV_NAME
```

## Training
```
python training.py
```

## Inference
```
python -m inference -m MODE -a CLASSIFIER_ARCHITECTURE -i INPUT_FOLDER -o OUTPUT_FOLDER
```

Flag | Description | Default value
-----|-----|-----
-m (or --mode) | Input image type: singlestack (sstack) or multistack (mstack) | sstack
-a (or --arch) | Classifier architecture | resnet50
-i (or --input) | Path to input folder | ./input/MODE
-o (or --output) | Path to output folder | ./output
