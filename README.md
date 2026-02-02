# Modality Gap–Driven Subspace Alignment Training Paradigm For Multimodal Large Language Models

<!-- <p align="center">
  📄 <a href="https://arxiv.org/abs/2503.22655">Paper</a> | 🤗 <a href="https://huggingface.co/datasets/Yu2020/Unicorn">Data</a>
</p> -->

# 📢 News

- [2026/02/02] 🏖️ **NOTE:** The author is currently on vacation and will address the To-Do items upon return.
- [2026/01/30] ✨ We release the **code** of the **ReVision**. Try training!
- [2025/06/10] ✨ We release the **code** of the **Unicorn**. Try training!
- [2025/04/15] ✨ Release **Unicorn-1.2M** & **Unicorn-Instruction-471K** Datasets. [[HF](https://huggingface.co/datasets/Yu2020/Unicorn)]


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

#### 3. Run Data Embed 

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


## Step 2: ReAlign

After generating the initial embeddings (Step 1), use this script to align the **Text Embeddings** into the **Image Embedding Space**. This process reduces the modality gap using the **ReAlign** method.

#### 1. How it Works
The script performs a robust, multi-pass statistical alignment:

1.  Global Mean: Calculates the precise global mean of both text and image vectors (using `float64` to prevent overflow).
2.  Trace Calculation: Computes the variance (Trace) of both modalities to determine the scaling factor:
    $$Scale = \sqrt{\frac{Trace_{img}}{Trace_{txt}}}$$
3.  Mean Correction: Estimates the shift required to align the centers of the distributions.
4.  Trace align: Applies the final transformation to the text embeddings:
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

Here is the **Training** section, written to perfectly match the style and context of your existing README. You can insert this section right after the **Step 2: ReAlign** section and before the **To Do** section.


# :rocket: Model Training

ReVision training is conducted using the processed embeddings Step 2.

### 1. Data Preparation

Ensure your workspace is organized with the necessary JSON annotations and the feature files generated from the ReAlign step.

```text
├── output/
│   └── aligned_feats/trace/              # [Input] The Aligned Embeddings (.pkl files) from Step 2
├── data/
│   ├── pretrain.json  # Pretraining JSON
│   └── sft.json                   # SFT JSON

```

### 2. Code Configuration (Crucial Step)

Before running the training scripts, you **must** manually configure the data loader to point to your feature directory.

1. Open the file: `bunny/util/data_utils.py`.
2. Locate the `LazySupervisedDataset` class.
3. Find the `folder_path` variable inside the `__init__` method.
4. **Modify it** to point to your aligned features directory (e.g., the absolute path to `output/aligned_feats/trace`).

**Example Modification in `bunny/util/data_utils.py`:**

```python
class LazySupervisedDataset(Dataset):
    def __init__(self, data_path, tokenizer, data_args):
        # ... (lines omitted)
        
        # [ACTION REQUIRED] Change this path to your Step 2 output directory
        # folder_path = '/old/path/...'
        folder_path = '/your/project/path/ReVision/output/aligned_feats/trace'
        
        # ... (The code will automatically load all .pkl files from this folder)

```

> **Note:** The data loader uses a dictionary to map IDs from your JSON dataset to the embeddings in the `.pkl` files. Ensure the `id` fields in your JSON match the IDs used during the Embedding Process.

### 3. Stage 1: Modality Substitution Pretraining

In this stage, we train the **Projector** (MLP) to align the language model with the synthetic visual embeddings.

```bash

```

**Run the command:**

```bash
cd ReVision

sh script/train/pretrain.sh

```

**Configuration Notes:**

* `--image_folder`: This argument in the script is **ignored** (or acts as a placeholder) because the code is hardcoded to load features from `folder_path` in `data_utils.py`.
* `--data_path`: Path to your pretraining JSON file.

### 4. Stage 2:  Visual Instruction Tuning

In this stage, we perform Visual Instruction Tuning on the LLM backbone.

**Run the command:**

```bash
sh script/train/finetune_full.sh

```

**Configuration Notes:**

* `--pretrain_mm_mlp_adapter`: **Must point to** the `mm_projector.bin` saved from Stage 1.
* `--data_path`: Path to your SFT JSON file.
* **Memory Optimization:** The script uses DeepSpeed ZeRO-3 (`zero3.json`) to handle the memory requirements of the 8B model.

# 📝 To Do

- [ ] Release embedding quality detection code.
- [ ] Refactor and optimize training code.
- [ ] Release **ReVision** Pre-training data.
- [ ] Release **ReVision** SFT data.
- [ ] Release evaluation benchmarks.
- [ ] Release **ReVision-8B** model weights.

# 🖊️ Citation

If you find this repository helpful, please consider citing our paper:

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
