# Grounded-Segment-Anything

[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/oEQYStnF2l8) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/automated-dataset-annotation-and-evaluation-with-grounding-dino-and-sam.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/camenduru/grounded-segment-anything-colab) [![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/IDEA-Research/Grounded-SAM) [![Replicate](https://replicate.com/cjwbw/grounded-recognize-anything/badge)](https://replicate.com/cjwbw/grounded-recognize-anything)  [![ModelScope Official Demo](https://img.shields.io/badge/ModelScope-Official%20Demo-important)](https://modelscope.cn/studios/tuofeilunhifi/Grounded-Segment-Anything/summary) [![Huggingface Demo by Community](https://img.shields.io/badge/Huggingface-Demo%20by%20Community-red)](https://huggingface.co/spaces/yizhangliu/Grounded-Segment-Anything) [![Stable-Diffusion WebUI](https://img.shields.io/badge/Stable--Diffusion-WebUI%20by%20Community-critical)](https://github.com/continue-revolution/sd-webui-segment-anything) [![Jupyter Notebook Demo](https://img.shields.io/badge/Demo-Jupyter%20Notebook-informational)](./grounded_sam.ipynb) [![Static Badge](https://img.shields.io/badge/GroundingDINO-arXiv-blue)](https://arxiv.org/abs/2303.05499) [![Static Badge](https://img.shields.io/badge/Segment_Anything-arXiv-blue)](https://arxiv.org/abs/2304.02643) [![Static Badge](https://img.shields.io/badge/Grounded_SAM-arXiv-blue)](https://arxiv.org/abs/2401.14159)


We plan to create a very interesting demo by combining [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment Anything](https://github.com/facebookresearch/segment-anything) which aims to detect and segment anything with text inputs! And we will continue to improve it and create more interesting demos based on this foundation. And we have already released an overall technical report about our project on arXiv, please check [Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks](https://arxiv.org/abs/2401.14159) for more details.

- 🔥 **[Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2)** is released now, which combines Grounding DINO with [SAM 2](https://github.com/facebookresearch/segment-anything-2) for any object tracking in open-world scenarios.
- 🔥 **[Grounding DINO 1.5](https://github.com/IDEA-Research/Grounding-DINO-1.5-API)** is released now, which is IDEA Research's **Most Capable** Open-World Object Detection Model!
- 🔥 **[Grounding DINO](https://arxiv.org/abs/2303.05499)** and **[Grounded SAM](https://arxiv.org/abs/2401.14159)** are now supported in Huggingface. For more convenient use, you can refer to [this documentation](https://huggingface.co/docs/transformers/model_doc/grounding-dino)

We are very willing to **help everyone share and promote new projects** based on Segment-Anything, Please check out here for more amazing demos and works in the community: [Highlight Extension Projects](#highlighted-projects). You can submit a new issue (with `project` tag) or a new pull request to add new project's links. 

![](./assets/grounded_sam_new_demo_image.png)

![](./assets/ram_grounded_sam_new.png)

**🍄 Why Building this Project?**

The **core idea** behind this project is to **combine the strengths of different models in order to build a very powerful pipeline for solving complex problems**. And it's worth mentioning that this is a workflow for combining strong expert models, where **all parts can be used separately or in combination, and can be replaced with any similar but different models (like replacing Grounding DINO with GLIP or other detectors / replacing Stable-Diffusion with ControlNet or GLIGEN/ Combining with ChatGPT)**.

## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

### Install with Docker

Open one terminal:

```
make build-image
```

```
make run
```

That's it.

If you would like to allow visualization across docker container, open another terminal and type:

```
xhost +
```


### Install without Docker (Haozheng's setup on G2)
You should set the environment variable manually as follows if you want to build a local GPU environment for Grounded-SAM:
```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-12.1/
```

Create environment on G2:

```bash
conda create -n groundedsam python=3.10
```

Install CUDA and Pytorch. Remember to change the cuda path to your own path. 


```jsx
conda install -y nvidia/label/cuda-12.1.0::cuda
conda install -y nvidia/label/cuda-12.1.0::cuda-cudart
```

```
conda env config vars set PATH=/home/hy648/.conda/envs/groundedsam/bin:$PATH -n groundedsam
conda env config vars set LD_LIBRARY_PATH=/home/hy648/.conda/envs/groundedsam/lib64:$LD_LIBRARY_PATH -n groundedsam
conda env config vars set CUDA_HOME=/home/hy648/.conda/envs/groundedsam -n groundedsam
conda env config vars set CPATH=/home/hy648/.conda/envs/groundedsam/targets/x86_64-linux/include:$CPATH -n groundedsam
```

```jsx
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

Install Grounding DINO:

```bash
pip install --no-build-isolation -e GroundingDINO
```


Install diffusers:

```bash
pip install --upgrade diffusers[torch]
```

Install RAM & Tag2Text:

```bash
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```

More details can be found in [install segment anything](https://github.com/facebookresearch/segment-anything#installation) and [install GroundingDINO](https://github.com/IDEA-Research/GroundingDINO#install) and [install OSX](https://github.com/IDEA-Research/OSX)


### Grounded-SAM: Detect and Segment Everything with Text Prompt

Here's the step-by-step tutorial on running `Grounded-SAM` demo:

**Step 1: Download the pretrained weights**

```bash
cd Grounded-Segment-Anything

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```


**Step 2: Running original grounded-sam demo**
```bash
# depends on your device 
export CUDA_VISIBLE_DEVICES=0
```

```python

python grounded_sam_depth.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/basket.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "basket" \
  --device "cuda"
```

The annotated results will be saved in `./outputs` as follows

<div align="center">

| Input Image | Annotated Image | Generated Mask |
|:----:|:----:|:----:|
| ![](./assets/demo1.jpg) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/grounded_sam/original_grounded_sam_demo1.jpg?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/grounded_sam/mask.jpg?raw=true) |

</div>



