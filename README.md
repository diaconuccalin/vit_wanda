# Adapted From the "Pruning LLMs by Weights and Activations" Paper

**A Simple and Effective Pruning Approach for Large Language Models** </br>
*Mingjie Sun\*, Zhuang Liu\*, Anna Bair, J. Zico Kolter* (* indicates equal contribution) <br>
Carnegie Mellon University, Meta AI Research and Bosch Center for AI <br>
[Paper](https://arxiv.org/abs/2306.11695) - [Project page](https://eric-mingjie.github.io/wanda/home.html)

```bibtex
@article{sun2023wanda,
  title={A Simple and Effective Pruning Approach for Large Language Models}, 
  author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
  year={2023},
  journal={arXiv preprint arXiv:2306.11695}
}
```

## Acknowledgement

Their repository is built upon the [SparseGPT](https://github.com/IST-DASLab/sparsegpt) repository.

## License

Their project is released under the MIT license.

# Pruning Image Classifiers

Adapted from the [solution](https://github.com/locuslab/wanda) offered by the authors, which is built on
the [dropout](https://github.com/facebookresearch/dropout) repository.

## Download Weights

Run the script [download_weights.sh](download_weights.sh) to download pretrained weights for ConvNeXt-B, DeiT-B and
ViT-B, which we used in the paper.

## Usage

Here is the command for pruning ConvNeXt/ViT models:

```
python main.py --model [ARCH] \
    --data_path [PATH to ImageNet] \
    --resume [PATH to the pretrained weights] \
    --prune_metric wanda \
    --prune_granularity row \
    --sparsity 0.5 
```

where:

- `--model`: network architecture, choices [`convnext_base`, `deit_base_patch16_224`, `vit_base_patch16_224`].
- `--resume`: model path to downloaded pretrained weights.
- `--prune_metric`: [`magnitude`, `wanda`].
- `--prune_granularity`: [`layer`, `row`].
