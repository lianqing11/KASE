# Known-class Aware Self-ensemble for Open Set Domain Adaptation

Pytorch implemention of our methdo for open set domain adaptation. Based on this implementation, our result is ranked 2nd in the [VisDA Challenge 2018](http://ai.bu.edu/visda-2018/).

## Enviorment
The code is developed under the following configuration.
#### Hardware:
1 GPU (with at least 11G GPU memories), which is set for the correspoinding batch size.

#### Software:
Python 3, Pytorch 0.4, and CUDA 8.0 is necessary before running the scripts. To install the required pythonn packages(expect Pytorch), run

```
pip install -r requirements.txt
```

## Datasets


Follow the github [REPO](https://github.com/VisionLearningGroup/visda-2018-public) to download the  [Syn2Real-O] dataset, and put it in the `./data/visda` folder.

## Training Examples

### Source only
```
python train_source_only.py --config cfgs/source_only_exp001.yaml
```

### Adabn
```
python train_adabn.py --config cfgs/adabn_exp001.yaml
```

### [MMD](https://arxiv.org/abs/1502.02791)
```
python train_mmd.py --config cfgs/mmd_exp001.yaml
```

### [BP](http://sites.skoltech.ru/compvision/projects/grl/)
```
python train_bp.py --config cfgs/bp_exp001.yaml
```

### [Self-Ensemble](https://arxiv.org/abs/1706.05208)
```
python train_se.py --config cfgs/se_exp001.yaml
```

### [KASE](https://arxiv.org/abs/1905.01068)
```
python train_kase.py --config cfgs/kase_exp001.yaml
```



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
