



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

## Step 1. Data Embed

#### prepare separate sets of image and text data (unpaired), formatted as follows:

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


## ⚖️ Step 2: ReAlign

After generating the initial embeddings (Step 1), use this script to align the **Text Embeddings** into the **Image Embedding Space**. This process reduces the modality gap using the **Trace (Variance) Matching** method.

#### 1. How it Works
The script performs a robust, multi-pass statistical alignment:

1.  Global Mean:** Calculates the precise global mean of both text and image vectors (using `float64` to prevent overflow).
2.  Trace Calculation:** Computes the variance (Trace) of both modalities to determine the scaling factor:
    $$Scale = \sqrt{\frac{Trace_{img}}{Trace_{txt}}}$$
3.  Mean Correction:** Estimates the shift required to align the centers of the distributions.
4.  Trace align:** Applies the final transformation to the text embeddings:
    $$X_{aligned} = (X_{text} - \mu_{text}) \cdot Scale + \mu_{image}$$

#### 2. Run Alignment
Execute `embed_ReAlign.py` to process the `.pkl` files generated in Step 1.

```bash
python embed_ReAlign.py \
    --input_dir "./output/text_feats" \
    --img_input_dir "./output/image_feats" \
    --output_dir "./output/aligned_feats" \
    --chunk_size 10000 \
    --strict_finite 1

```

**Arguments:**

* `--input_dir`: Path to the folder containing **Text** `.pkl` files (from Step 1).
* `--img_input_dir`: Path to the folder containing **Image** `.pkl` files.
* `--output_dir`: Where to save the aligned text embeddings.
* `--chunk_size`: Number of vectors to process in memory at once (default: 10,000).
* `--strict_finite`: Set to `1` (default) to immediately abort if `NaN` or `Inf` values are detected.

#### 3. Output

The script creates a `trace/` subdirectory inside your output folder.

* **Aligned Text:** `output/aligned_feats/trace/text_embeds_X_trace.pkl`
* **Statistics:** `output/aligned_feats/trace_stats.pkl` (Contains calculated means, scale factor, and trace values for validation).

> **Note:** This script only transforms the **Text** embeddings. The **Image** embeddings remain unchanged as they serve as the "anchor" distribution.

```


# Citation
If you find this repository helpful, please cite the paper below.

```bibtex
@article{yu2025unicorn,
  title={Unicorn: Text-only data synthesis for vision language model training},
  author={Yu, Xiaomin and Ding, Pengxiang and Zhang, Wenjie and Huang, Siteng and Gao, Songyang and Qin, Chengwei and Wu, Kejian and Fan, Zhaoxin and Qiao, Ziyue and Wang, Donglin},
  journal={arXiv preprint arXiv:2503.22655},
  year={2025}
}
```

# 📧 Contact

If you have any questions or are interested in collaborating, please reach out: **xmyu02@gmail.com**
