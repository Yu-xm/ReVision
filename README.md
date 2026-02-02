



# Modality Gap–Driven Subspace Alignment Training Paradigm For Multimodal Large Language Models

<!-- <p align="center">
  📄 <a href="https://arxiv.org/abs/2503.22655">Paper</a> | 🤗 <a href="https://huggingface.co/datasets/Yu2020/Unicorn">Data</a>
</p> -->

## News

- [2026/01/30] 🔥 We release the **code** of the **ReVision**. Try training!
- [2025/06/10] 🔥 We release the **code** of the **Unicorn**. Try training!
- [2025/04/15] 🔥 Release **Unicorn-1.2M** & **Unicorn-Instruction-471K** Datasets. [[HF](https://huggingface.co/datasets/Yu2020/Unicorn)]

## Training

**Our code is based on** [Bunny](https://github.com/BAAI-DCAI/Bunny). Thanks!

# :gear: Env

Create a conda virtual environment and activate it:

  ```shell
  conda create -n ReVision python=3.10
  conda activate ReVision
  ```

Basic requirements

  ```shell
  pip install --upgrade pip  
  pip install transformers=4.44.0
  pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu124
  ```

Install flash-attention

  ```shell
  pip install packaging
  pip install flash-attn --no-build-isolation
  ```
# :hammer_and_wrench: Embedding Process

### Process your own data as follows:

First, prepare separate sets of image and text data (unpaired), formatted as follows:

```
python image_embed.py
```

Mean shift to get synthetic image embeddings

```
python embed_mean.py
```

Then, change the embedding path in `data_utils.py`

```
folder_path = ''
```
Note: the same pkl file is used in both the pretraining and instruction-tuning stages

### Pretrain

```
sh script/train/pretrain.sh
```

### Instruction Tuning

```
sh script/train/finetune_full.sh
```

## Citation
If you find this repository helpful, please cite the paper below.

```bibtex
@article{yu2025unicorn,
  title={Unicorn: Text-only data synthesis for vision language model training},
  author={Yu, Xiaomin and Ding, Pengxiang and Zhang, Wenjie and Huang, Siteng and Gao, Songyang and Qin, Chengwei and Wu, Kejian and Fan, Zhaoxin and Qiao, Ziyue and Wang, Donglin},
  journal={arXiv preprint arXiv:2503.22655},
  year={2025}
}
```

## Contact

If you have any questions, please contact: xmyu02@gamil.com
