# ViStruct: Visual Structural Knowledge Extraction via Curriculum Guided Code-Vision Representation
Code and data of the EMNLP 2023 [paper](https://arxiv.org/abs/2311.13258) "ViStruct: Visual Structural Knowledge Extraction via Curriculum Guided Code-Vision Representation"

# Environment
```bash
git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
pip install OFA/transformers/
conda env create -f environment.yml
```

# Curriculum Guided Pretraining
## Data Preparation
### Download the ImageCode data
```bash
bash data/download.sh
```

## Train model
```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/model/train.py configs/base-concept.yml
```



# Citation

Please kindly cite our [paper](https://arxiv.org/abs/2311.13258):

```
@article{chen2023vistruct,
  title={ViStruct: Visual Structural Knowledge Extraction via Curriculum Guided Code-Vision Representation},
  author={Chen, Yangyi and Wang, Xingyao and Li, Manling and Hoiem, Derek and Ji, Heng},
  journal={arXiv preprint arXiv:2311.13258},
  year={2023}
}
```
