# aiMotive Traffic Sign & Traffic Light Dataset Loader

## Download
The dataset can be downloaded from this [repository](https://github.com/aimotive/aimotive_tl_ts_dataset).

## Installation
The repository has been tested on Ubuntu with Python 3.10. Currently no Windows support is available.
### Create a conda environment
```
conda create --name aimdataset python=3.10
conda activate aimdataset
```

### Clone repository and checkout
```
git clone https://github.com/aimotive/aimotive-dataset-loader.git
cd aimotive-dataset-loader
git checkout aimotive-tlts-dataset-loader
```

### Install requirements
```
pip install -r requirements.txt
```

## Examples
The repository includes a small sample dataset with 227 keyframes. The examples demonstrate how the data can be rendered
and loaded to PyTorch framework.

### Run rendering example
```
PYTHONPATH=$PYTHONPATH: python examples/example_render.py --object-type traffic_light
```
or
```
PYTHONPATH=$PYTHONPATH: python examples/example_render.py --object-type traffic_sign
```

### Run PyTorch loader example
#### Install torch
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Run script
```
PYTHONPATH=$PYTHONPATH: python examples/pytorch_loader.py
```

### Cite our work
If you use this code or aiMotive 3D Traffic Light and Traffic Sign Dataset in your research, please cite our [work](https://arxiv.org/abs/2409.12620v1) by using the following BibTeX entries:

```latex
@article{kunsagi2024aimotive,
  title={Accurate Automatic 3D Annotation of Traffic Lights and Signs for Autonomous Driving},
  author={Kuns{\'a}gi-M{\'a}t{\'e}, S{\'a}ndor and Pet{\H{o}}, Levente and Seres, Lehel and Matuszka, Tam{\'a}s},
  booktitle={European Conference on Computer Vision 2024 Workshop on Vision-Centric Autonomous Driving}
}

@misc{kunságimáté2024accurateautomatic3dannotation,
      title={Accurate Automatic 3D Annotation of Traffic Lights and Signs for Autonomous Driving}, 
      author={Sándor Kunsági-Máté and Levente Pethő and Lehel Seres and Tamás Matuszka},
      year={2024},
      eprint={2409.12620},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.12620}, 
}
```
