<!-- <h1 align="center">Modality Gap–Driven Subspace Alignment Training Paradigm For Multimodal Large Language Models</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Theory-Modality_Gap-C92D39" alt="Theory">
  <img src="https://img.shields.io/badge/Method-Modality_Alignment-C92D39" alt="Method">
  <img src="https://img.shields.io/badge/Model-MLLMs-007EC6" alt="Model">
</p>



<img src="./asset/ReAlign.png" alt="Logo">

<font size=7><div align='center' > [[📖 Paper](https://github.com/Yu-xm/ReVision.git)] [[📊 Dataset](https://github.com/Yu-xm/ReVision.git)] [[🌐 Model](https://github.com/Yu-xm/ReVision.git)] </div></font> -->

<div align="center">

# Modality Gap–Driven Subspace Alignment Training <br> Paradigm For Multimodal Large Language Models

<p align="center">
  <a href="https://github.com/Yu-xm/ReVision">
    <img src="https://img.shields.io/badge/Theory-Modality_Gap-C92D39?style=for-the-badge" alt="Theory">
  </a>
  <a href="https://github.com/Yu-xm/ReVision">
    <img src="https://img.shields.io/badge/Method-ReAlign-C92D39?style=for-the-badge" alt="Method">
  </a>
  <a href="https://github.com/Yu-xm/ReVision">
    <img src="https://img.shields.io/badge/Model-MLLMs-007EC6?style=for-the-badge" alt="Model">
  </a>
</p>

<p align="center">
  <a href="https://github.com/Yu-xm/ReVision">
    <img src="https://img.shields.io/badge/📖-Paper-ff69b4?style=flat-square&logo=arxiv&logoColor=white" alt="Paper">
  </a>
  <a href="https://github.com/Yu-xm/ReVision">
    <img src="https://img.shields.io/badge/🤗-HuggingFace_Dataset-ffbd45?style=flat-square&logo=huggingface&logoColor=white" alt="Dataset">
  </a>
  <a href="https://github.com/Yu-xm/ReVision">
    <img src="https://img.shields.io/badge/🌐-Model_Zoo-blue?style=flat-square&logo=github&logoColor=white" alt="Model">
  </a>
</p>
</div>

<div align="center">
<img src="./asset/ReAlign.png" alt="Logo" width="90%">
</div>

<p align="center">
  <a href="#core-insight">💡 Core Insight</a> •
  <a href="#env">🛠️ Env Installation</a> •
  <a href="#process">⚙️ Data Process</a> •
  <a href="#training">🚀 Training</a>
</p>

<hr>

# 📢 News

- **`2026.02.09`** 💡 **Highlight:** Please check the [Core Insight](#core-insight) section! We explain how ReAlign enables **efficient scaling** by decoupling dependence on multimodal paired data.
- **`2026.02.02`** 🏖️ **NOTE:** The author is currently on a journey and will address the To-Do items upon return.
- **`2026.02.02`** ✨ We release the code of the **ReVision**. Try training!
- **`2026.01.31`** 📝 We release the paper ReVision.
- **`2025.06.10`** ✨ We release the code of the **Unicorn**. Try training!
- **`2025.04.15`** 🗂️ We release **Unicorn-1.2M** & **Unicorn-Instruction-471K** Datasets. [[HF](https://huggingface.co/datasets/Yu2020/Unicorn)]
- **`2025.03.28`** 📝 We release the paper Unicorn. [[PAPER](https://arxiv.org/abs/2503.22655)]


# 💡 Must Read!!! <a id="core-insight"></a>

> [!IMPORTANT]
> **<div align="center">Core Insight</div>**
>
> <div align="justify">
>
> ReAlign leverages the Modality Gap phenomenon within the high-dimensional hyperspherical embedding space of multimodal contrastive learning to precisely map unpaired text representations into the visual representation distribution. Within this framework, text serves a dual role: the transformed text embeddings function as pseudo-visual representations providing equivalent semantics, while the original text serves as the **supervision signal** for the MLLM. This modality substitution mechanism decouples the model's dependence on expensive paired data, thereby serving as an efficient scaling law strategy that can be seamlessly integrated throughout the entire training lifecycle (spanning Pre-training, SFT, and RL). Theoretically, this paradigm is not limited to image-text pairs but can be generalized to *any* modalities possessing an aligned subspace.
>
> </div>

# :gear: Env <a id="env"></a>

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

# :hammer_and_wrench: Embedding Process <a id="process"></a>

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

#### 1. How it Works: The ReAlign Strategy

This script implements **ReAlign**, a robust, training-free statistical alignment strategy derived from our *Fixed-frame Modality Gap Theory*. It bridges the geometric misalignment by mapping text representations ($e_y$) into the visual distribution ($e_x$) through a precise three-stage process:

1.  **Anchor Alignment.**
    We first address the first-order distributional shift by eliminating the mean difference. The source embeddings are centered and then shifted to the target anchor:
    $$\dot{e}_y = (e_y - \mu_y) + \mu_x$$

2.  **Trace Alignment.**
    Next, we adjust the scale of residuals to match the global energy of the visual modality while preserving the spectral structure. A scaling factor $s$ is derived from the global trace (variance):
    $$s = \sqrt{\frac{Trace_{img}}{Trace_{txt}}} \quad \Rightarrow \quad \tilde{e}_y = \mu_x + s(e_y - \mu_y)$$

3.  **Centroid Alignment (Phantom Drift Correction).**
    While affine transformations align statistics in Euclidean space, the subsequent spherical projection induces a secondary "Phantom Drift." We perform a final centroid correction on the unit hypersphere to strictly align the angular mass centers:
    $$e''_y = e'_y - \mu' + \mu_x$$

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


# :rocket: Training <a id="training"></a>

ReVision training is conducted using the processed embeddings in Step 2.

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

In this stage, we train the projector (MLP) to align the language model with the synthetic visual embeddings.

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

If you have any questions or are interested in collaborating, please reach out: **yuxm02@gmail.com**
