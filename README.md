



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

#### 1. Data Preparation
Format your `dataset.json` as a dictionary containing two separate lists: `images` and `texts`.
* **images:** A list of dictionaries, each with an `id` and `image` path.
* **texts:** A list of dictionaries, each with an `id` and `text` content.

```json
{
  "images": [
    {
      "id": "img_001",
      "image": "0001.jpg"
    },
    {
      "id": "img_002",
      "image": "folder/0002.png"
    }
  ],
  "texts": [
    {
      "id": "txt_001",
      "text": "This is a text sample description."
    },
    {
      "id": "txt_002",
      "text": "Another independent text entry."
    }
  ]
}

```

#### 2. Directory Structure

Ensure your directory looks similar to this before running:

```text
├── data/
│   ├── images/             # Root folder for images
│   └── dataset.json        # The JSON index file above
├── models/
│   ├── llm2clip-openai/    # Local vision encoder path
│   └── llm2vec-llama3/     # Local text encoder path
└── embed.py

```

#### 3. Run Feature Extraction

Run the script to generate embeddings. By default, this script runs **Offline** (using local model paths).

```bash
python embed.py \
    --data_json "./data/dataset.json" \
    --image_root "./data/images" \
    --output_text_dir "./output/text_feats" \
    --output_image_dir "./output/image_feats" \
    --llm2clip_path "/path/to/local/llm2clip-model" \
    --llm_model_name "/path/to/local/llm2vec-model" \
    --bsz 512 \
    --modality both

```

**Arguments:**

* `--modality`: Choose `both`, `text`, or `image`.
* `--bsz`: Batch size (default 1024; reduce to 512 or 256 if OOM occurs).
* `--online`: Add this flag if you want to allow Hugging Face Hub access.

#### 4. Output

The script saves features in chunked `.pkl` files (default 200k records per file).

* `output/text_feats/text_embeds_1.pkl`
* `output/image_feats/image_embeds_1.pkl`

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
