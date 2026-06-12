<p align="center">
  <h1 align="center">Ex-Omni</h1>
  <p align="center"><b>Ex-Omni is a omni-modal framework for generating response text, speech audio, and 3D facial animation from text or speech input.</b></p>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2602.07106v2">
    <img src="https://img.shields.io/badge/arXiv-2602.07106-b31b1b.svg?logo=arxiv" alt="arXiv 2602.07106" />
  </a>
  <a href="https://huggingface.co/Tencent/Ex-Omni">
    <img src="https://img.shields.io/badge/Hugging%20Face-Tencent%2FEx--Omni-yellow?logo=huggingface" alt="CKPT" />
  </a>
  <a href="./LICENSE.txt">
    <img src="https://img.shields.io/badge/License-See%20LICENSE-blue.svg" alt="License" />
  </a>
</p>

---

`Ex-Omni` is an public release for omni-modal response generation. Given text or speech input, the system can produce response text, speech units / decoded audio, and 52-dimensional facial blendshape coefficients, with optional rendering into a talking-face video.

## 📖 Table of Contents
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Checkpoints and Assets](#checkpoints-and-assets)
  - [Launch the Demo](#launch-the-demo)
- [Acknowledgements](#-acknowledgements)
- [Citation](#-citation)

## 🗂️ Repository Structure

```text
.
├── asset/                              # Download the mesh templates here
├── ckpt/                               # Download the checkpoints here
├── cosyvoice/                          # Runtime audio decoder modules
├── deploy.py                           # Main Gradio entrypoint
├── deploy_base.py                      # Shared inference pipeline and UI logic
├── ex_omni/
│   ├── constants.py                    # Runtime constants
│   ├── flow_inference.py               # Audio decoder wrapper
│   ├── render_utils.py                 # Blendshape rendering utilities
│   └── model/
│       ├── language_model/             # Omni model wrapper
│       ├── speech_encoder/             # Whisper speech encoder
│       ├── speech_projector/           # Speech projector
│       ├── speech_generator/           # Speech generator
│       └── blendshape_generator/       # Blendshape generator
├── requirements.txt                    # Python dependencies
└── LICENSE.txt                         # License file
```

## ⚙️ Quick Start

### Installation
```bash
# 1. Create and activate environment
conda create -n Ex-Omni python=3.10 -y
conda activate Ex-Omni

# 2. Install PyTorch (example: CUDA 12.6)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu1216

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Install pytorch3d separately according to your CUDA / PyTorch version
# See official pytorch3d installation instructions
```

### Checkpoints and Assets

Prepare the following assets before running inference:

| Component | Expected Path | Source |
|:--|:--|:--|
| Ex-Omni checkpoint | `ckpt/Ex-Omni/` | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Tencent%2FEx--Omni-yellow?logo=huggingface)](https://huggingface.co/Tencent/Ex-Omni) |
| Flow decoder checkpoint | `ckpt/glm-4-voice-decoder/flow.pt` | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-zai--org%2Fglm--4--voice--decoder-yellow?logo=huggingface)](https://huggingface.co/zai-org/glm-4-voice-decoder) |
| HiFT decoder checkpoint | `ckpt/glm-4-voice-decoder/hift.pt` | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-zai--org%2Fglm--4--voice--decoder-yellow?logo=huggingface)](https://huggingface.co/zai-org/glm-4-voice-decoder) |
| EmoTalk mesh template | `asset/EmoTalk.npz` | [![GitHub](https://img.shields.io/badge/GitHub-X--niper%2FUniTalker-181717?logo=github)](https://github.com/X-niper/UniTalker) |
| Claire mesh template | `asset/claire.npz` | [![Hugging Face Datasets](https://img.shields.io/badge/Hugging%20Face%20Datasets-NVIDIA%2FAudio2Face--3D--Dataset--v1.0.0--claire-yellow?logo=huggingface)](https://huggingface.co/datasets/Nvidia/Audio2Face-3D-Dataset-v1.0.0-claire) |

> For EmoTalk template, please download the `resources.zip` from UniTalker and extract the `EmoTalk.npz` file. For Claire template, please obtain the original Claire asset yourself and convert it into the `.npz` format.


### Launch the Demo

```bash
python deploy.py \
  --model-path ckpt/Ex-Omni \
  --flow_ckpt_path ckpt/glm-4-voice-decoder/flow.pt \
  --hift_ckpt_path ckpt/glm-4-voice-decoder/hift.pt \
  --template_type emotalk \
  --port 8080
```

Then open:

```text
http://localhost:8080
```

## 🙏 Acknowledgements

We would like to thank the authors of [OpenOmni](https://github.com/RainBowLuoCS/OpenOmni), [LLaMA-Omni2](https://github.com/ictnlp/LLaMA-Omni2), [EmoTalk](https://github.com/psyai-net/EmoTalk_release) and [UniTalker](https://github.com/X-niper/UniTalker). Parts of the implementation and overall system design were developed with reference to their open-source release.

## 📄 Citation

If you use this project, please cite our paper:

```bibtex
@misc{zhang2026exomnienabling3dfacial,
      title={Ex-Omni: Enabling 3D Facial Animation Generation for Omni-modal Large Language Models}, 
      author={Haoyu Zhang and Zhipeng Li and Yiwen Guo and Tianshu Yu},
      year={2026},
      eprint={2602.07106},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.07106}, 
}
```
