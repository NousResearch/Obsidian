# Obsidian: Multimodal LLM for Everyone

<p align="center">
  <img src="images/obsidian.png" width="50%"> <br>
</p>

<p align="center">
  <a href="https://huggingface.co/NousResearch/Obsidian-3B-V0.5">🤗 Model</a> •
  <a href="https://colab.research.google.com/drive/1C1FkoeZYBv3dZELaKgxahoZzWPfz0En8?usp=sharing">📓 Colab Demo</a> •
  <a href="#quick-start">🚀 Quick Start</a> •
  <a href="#model-capabilities">✨ Capabilities</a>
</p>

<p align="center">Obsidian is a joint work between <a href="https://huggingface.co/NousResearch">Nous Research</a> and <a href="https://huggingface.co/vilm">Virtual Interactive</a>. Special thanks to <b>LDJ</b> and <b>qnguyen3</b> for making this work possible.</p>

---

## Table of Contents

- [Model Capabilities](#model-capabilities)
- [Quick Start](#quick-start)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Setup](#step-by-step-setup)
- [Usage](#usage)
  - [Web Demo UI](#web-demo-ui)
  - [Python API](#python-api)
  - [Command Line Inference](#command-line-inference)
- [Training](#training)
  - [Pretraining](#1-pretraining)
  - [Instructional Finetuning](#2-instructional-finetuning)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Acknowledgement](#acknowledgement)

---

## Model Capabilities

Obsidian is a powerful multimodal language model that combines vision and language understanding. Here's what it can do:

| Capability | Description |
|------------|-------------|
| 🖼️ **Image Understanding** | Analyze and describe images in detail |
| 💬 **Visual Q&A** | Answer questions about image content |
| 📝 **Image Captioning** | Generate accurate descriptions of visual content |
| 🔍 **OCR & Text Recognition** | Read and interpret text within images |
| 🎨 **Visual Reasoning** | Perform complex reasoning about visual scenes |
| 📊 **Chart/Diagram Analysis** | Interpret charts, graphs, and diagrams |

### Model Variants

| Model | Parameters | HuggingFace |
|-------|------------|-------------|
| Obsidian-3B-V0.5 | 3B | [NousResearch/Obsidian-3B-V0.5](https://huggingface.co/NousResearch/Obsidian-3B-V0.5) |

---

## Quick Start

**Try it instantly on Google Colab:** [Open Colab Demo](https://colab.research.google.com/drive/1C1FkoeZYBv3dZELaKgxahoZzWPfz0En8?usp=sharing)

> **Note:** After opening the Gradio interface, give the model about 2 minutes to load, then refresh the page.

For local installation, continue to the [Installation](#installation) section below.

---

## Installation

### Prerequisites

Before installing Obsidian, ensure you have:

- **Python** 3.8 or higher (3.10 recommended)
- **CUDA** 11.7+ with compatible NVIDIA GPU (8GB+ VRAM recommended)
- **Git** for cloning the repository
- **Conda** (recommended) or pip for package management

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/NousResearch/Obsidian.git
cd Obsidian
```

#### 2. Create and Activate Environment

Using Conda (recommended):
```bash
conda create -n obsidian python=3.10 -y
conda activate obsidian
```

Or using venv:
```bash
python -m venv obsidian-env
source obsidian-env/bin/activate  # Linux/Mac
# obsidian-env\Scripts\activate   # Windows
```

#### 3. Install Core Dependencies

```bash
pip install --upgrade pip
pip install -e .
```

#### 4. Install Training Dependencies

Required for both training and inference:
```bash
pip install ninja
pip install flash-attn --no-build-isolation
```

#### 5. Download the Multimodal Projector

```bash
bash scripts/download_mm_projector.sh
```

#### 6. Install Compatible Transformers Version

```bash
pip install --upgrade transformers==4.34.0
```

### Verify Installation

```bash
python -c "from llava.model import LlavaLlamaForCausalLM; print('✓ Obsidian installed successfully!')"
```

---

## Usage

### Web Demo UI

The easiest way to interact with Obsidian is through the web interface. You'll need three terminal windows:

#### Terminal 1: Launch the Controller
```bash
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

#### Terminal 2: Launch the Gradio Web Server
```bash
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```

#### Terminal 3: Launch the Model Worker
```bash
python -m llava.serve.model_worker \
    --host 0.0.0.0 \
    --controller http://localhost:10000 \
    --port 40000 \
    --worker http://localhost:40000 \
    --model-path NousResearch/Obsidian-3B-V0.5
```

Wait for "Uvicorn running on ..." to appear, then refresh your browser. The model will now be available in the Gradio interface.

### Python API

Use Obsidian directly in your Python code:

```python
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import AutoTokenizer
from PIL import Image
import torch

# Initialize
disable_torch_init()
model_path = "NousResearch/Obsidian-3B-V0.5"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = LlavaLlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load and process image
image = Image.open("your_image.jpg")

# Create conversation
conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], "<image>\nDescribe this image in detail.")
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# Generate response
# (Add your inference code here based on the model's generate method)
```

### Command Line Inference

For quick single-image inference:

```bash
python -m llava.serve.cli \
    --model-path NousResearch/Obsidian-3B-V0.5 \
    --image-file "path/to/your/image.jpg"
```

---

## Training

### 1. Pretraining

Download the 558K subset of the LAION-CC-SBU dataset with BLIP captions from [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

**Training Time:** ~2.5 hours on 4x A100 (80G) at 336px resolution.

Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/pretrain.sh)

Key parameters:
- `--mm_projector_type mlp2x_gelu`: Two-layer MLP vision-language connector
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px

### 2. Instructional Finetuning

#### Download Required Data

1. **Annotations:** [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)

2. **Images from source datasets:**
   - COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
   - GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
   - OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing) (save all files as `.jpg`)
   - TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
   - VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

#### Organize Data Structure

```
./playground/data/
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

---

## Evaluation

### GPT-assisted Evaluation

Our evaluation pipeline provides comprehensive assessment of vision-language capabilities.

#### Step 1: Generate Model Responses

```bash
python model_vqa.py \
    --model-path ./checkpoints/LLaVA-13B-v0 \
    --question-file playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder /path/to/coco2014_val \
    --answers-file /path/to/answer-file-our.jsonl
```

#### Step 2: Evaluate Against Reference

```bash
OPENAI_API_KEY="your-api-key" python llava/eval/eval_gpt_review_visual.py \
    --question playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --context llava/eval/table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list /path/to/answer-file-ref.jsonl /path/to/answer-file-our.jsonl \
    --rule llava/eval/table/rule.json \
    --output /path/to/review.json
```

#### Step 3: Summarize Results

```bash
python summarize_gpt_review.py
```

### ScienceQA Benchmark

See the [ScienceQA documentation](https://github.com/haotian-liu/LLaVA/blob/main/docs/ScienceQA.md) for evaluation instructions.

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **CUDA out of memory** | Reduce batch size or use `--load-in-8bit` flag |
| **flash-attn installation fails** | Ensure CUDA toolkit matches PyTorch CUDA version |
| **Model not appearing in Gradio** | Wait for "Uvicorn running..." message, then refresh |
| **Slow inference** | Ensure you're using GPU (`device_map="auto"`) |
| **Import errors** | Run `pip install -e .` from the Obsidian directory |

### GPU Memory Requirements

| Model | Minimum VRAM | Recommended VRAM |
|-------|--------------|------------------|
| Obsidian-3B-V0.5 | 6GB | 8GB+ |
| Obsidian-3B-V0.5 (8-bit) | 4GB | 6GB+ |

---

## Acknowledgement

### Original LLaVA Project

*Visual instruction tuning towards large language and vision models with GPT-4 level capabilities.*

[[Project Page](https://llava-vl.github.io/)] [[Demo](https://llava.hliu.cc/)] [[Data](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)] [[Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)]

**Improved Baselines with Visual Instruction Tuning** [[Paper](https://arxiv.org/abs/2310.03744)]
[Haotian Liu](https://hliu.cc), [Chunyuan Li](https://chunyuan.li/), [Yuheng Li](https://yuheng-li.github.io/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/)

**Visual Instruction Tuning** (NeurIPS 2023, **Oral**) [[Paper](https://arxiv.org/abs/2304.08485)]
[Haotian Liu*](https://hliu.cc), [Chunyuan Li*](https://chunyuan.li/), [Qingyang Wu](https://scholar.google.ca/citations?user=HDiw-TsAAAAJ&hl=en/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/) (*Equal Contribution)

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
