# Mesenchymal Stem Cells Classification and Detection

## Summary
This repository contains the source of our paper, currently under review by the [International Journal of Stem Cells](https://www.ijstemcell.com/main.html).

## Dataset
This [dataset](https://doi.org/10.6084/m9.figshare.26367562) comprises microscope images of Mesenchymal Stem Cells cultured in single-layer and multi-layer flasks. These images are categorized into four classes according to the cell confluence level.
+ Class 1: 0-40%
+ Class 2: 40-60%
+ Class 3: 60-80%
+ Class 4: 80-100%
Furthermore, single-stack images contain bounding box annotations that indicate the presence of abnormal cells.

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

## Acknowledgements
The RetinaNet module used is from the [PyTorch RetinaNet implementation](https://github.com/yhenon/pytorch-retinanet.git).
