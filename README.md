# NVSA 

### Michael Hersche, Mustafa Zeqiri, Luca Benini, Abu Sebastian, Abbas Rahimi

[![DOI](https://zenodo.org/badge/587315616.svg)](https://zenodo.org/badge/latestdoi/587315616)

_Nature Machine Intelligence'23_

This library implements the NVSA functionalities of the paper "A Neuro-vector-symbolic Architecture for Solving Raven's Progressive Matrices". A complete example for solving RAVEN can be found [[here]](https://github.com/IBM/neuro-vector-symbolic-architectures-raven) 



## [[Preprint]](https://arxiv.org/pdf/2203.04571.pdf)

## Module Installation

This module can be installed using `pip`, and then simply used by
`import nvsa`.

As a prerequisite you should have already installed `torch` and `numpy`. 

### From GitHub

To install module from GitHub use one of:
```bash
# Install using ssh
pip install git+git@github.com:IBM/neuro-vector-symbolic-architectures.git
# Install using https
pip install git+https://github.com:IBM/neuro-vector-symbolic-architectures.git
```

If you already have pytorch installed you might want to additionally use `--no-dependencies` flag:
```bash
# Install using ssh, without dependencies
pip install git+git@github.com:IBM/neuro-vector-symbolic-architectures.git --no-dependencies
# Install using ssh, without dependencies
pip install git+https://github.com:IBM/neuro-vector-symbolic-architectures.git --no-dependencies
```

### From local repository

This version will automatically reflect any code changes that you make.

```bash
# Install from editable local version
pip install -e /path/to/local/neuro-vector-symbolic-architectures
# Also possible without dependencies
pip install -e /path/to/local/neuro-vector-symbolic-architectures --no-dependencies
```

## Citation

If you use the work released here for your research, please cite the preprint of this paper:
```
@article{hersche2022neuro,
  title={A Neuro-vector-symbolic Architecture for Solving Raven's Progressive Matrices},
  author={Hersche, Michael and Zeqiri, Mustafa and Benini, Luca and Sebastian, Abu and Rahimi, Abbas},
  journal={arXiv preprint arXiv:2203.04571},
  year={2022}
}
```

## License
Our code is licensed under Apache 2.0. Please refer to the LICENSE file for the licensing of our code. 
