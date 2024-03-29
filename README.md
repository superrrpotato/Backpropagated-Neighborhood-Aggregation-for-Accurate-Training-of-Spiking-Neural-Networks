# Backpropagated Neighborhood Aggregation for Accurate Training of Spiking Neural Networks (NA)

This repository is the official implementation of Backpropagated Neighborhood Aggregation for Accurate Training of Spiking Neural Networks

<img src="Poster-1 2.png" alt="Poster" width="950"/>

## Citation

```bibtex

@inproceedings{yang2021backpropagated,
  title={Backpropagated Neighborhood Aggregation for Accurate Training of Spiking Neural Networks},
  author={Yang, Yukun and Zhang, Wenrui and Li, Peng},
  booktitle={International Conference on Machine Learning},
  pages={11852--11862},
  year={2021},
  organization={PMLR}
}

```


## Requirements
### Dependencies and Libraries
* python 3.7
* pytorch
* torchvision

### Installation
To install requirements:

```setup
pip install -r requirements.txt
```

### Datasets
NMNIST: [dataset](https://www.garrickorchard.com/datasets/n-mnist), [preprocessing](https://github.com/stonezwr/TSSL-BP/tree/master/preprocessing/NMNIST)

## Training
### Before running
Modify the data path and network settings in the [config files](https://github.com/superrrpotato/Backpropagated-Neighborhood-Aggregation-for-Accurate-Training-of-Spiking-Neural-Networks/tree/master/Networks). 

Select the index of GPU in the [main.py](https://github.com/superrrpotato/Backpropagated-Neighborhood-Aggregation-for-Accurate-Training-of-Spiking-Neural-Networks/blob/master/main.py#L234) (0 by default)

### Run the code
```sh
$ python main.py -config Networks/config_file.yaml
$ python main.py -config Networks/config_file.yaml -checkpoint checkpoint/ckpt.pth // load the checkpoint
```
