
<p align="center">
  <img src="https://imgur.com/jA2mxjt.png" width="320" alt="MLOPS">
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.6%2B-blue?logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/NumPy-<2-013243?logo=numpy&logoColor=white" />
    <img src="https://img.shields.io/badge/Pandas-latest-150458?logo=pandas&logoColor=white" />
    <img src="https://img.shields.io/badge/OpenCV-opencv--python-5C3EE8?logo=opencv&logoColor=white" />
    <img src="https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white" />
    <img src="https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white" />
    <img src="https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?logo=huggingface&logoColor=black" />
    <img src="https://img.shields.io/badge/Datasets-HuggingFace-FFD21E?logo=huggingface&logoColor=black" />
    <img src="https://img.shields.io/badge/Accelerate-HuggingFace-FFD21E?logo=huggingface&logoColor=black" />
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" />
</p>


# PyYel Deep Learning

The PyYel Machine Learning Operations extension. This repository regroups tools to simplify and speed up Python AI solutions conception process.

## Quick start
1. Clone the repository and Install the library.

``` bash
your_path/PyYel-MLOps> pip install .
```

2. Import the library into you code.

``` python
import pylml
```

3. Import the relevant features.

``` python
from pylml.LLM import LLMDecodingPhi, LLMEncodingBARTLargeMNLI
from pylml.CNN import CNNClassificationResNet
```

## Content

The content of pylml. Unless specified diferently, all the modules may be directly imported into Python code as libraries.

### CNN (Convolutional Neural Networks)

|Source model|PyYel model|Task|Status|
|------------|-----------|----|------|
|ResNet|CNNCLassificationResNet|Classification|Implemented|
|FasterRCNN|CNNDetectionFasterRCNN|Detection|Implemented|
|SSD|CNNDetectionSSD|Detection|Implemented|
|RetinaNet|CNNDetectionRetinaNet|Detection|TODO|
|/|CNNKeypoint|Keypoint detection|TODO|
|FCN|CNNSegmentationFCN|Segmentation|Implemented/TODO|
|DeeplabV3|CNNSegmentationDeeplabV3|Segmentation|Implemented/TODO|

**Note:** _Traditionnal computer vision networks. Features a model builder to design custom small-sized networks._

### FCN (Fully Connected Networks)

|Source model|PyYel model|Task|Status|
|------------|-----------|----|------|
|/|FCNBuilder|/|TODO|

**Note:** _Dense models. Features a model builder to design custom small-sized networks._

### LLM (Large Language Models)

|Source model|PyYel model|Task|Status|
|------------|-----------|----|------|
|Mistral7B v0.1|LLMDecodingMistral7B|Decoding: text-to-text generation|Implemented|
|OPT 125M|LLMDecodingOPT125m|Decoding: text-to-text generation|Implemented|
|Phi 3.5 Mini Instruct|LLMDecodingPhi|Decoding: text-to-text generation|Implemented|
|Phi 3.5 MoE|LLMDecodingPhiMoE|Decoding: text-to-text generation|Implemented/TODO|
|BART Large|LLMEncodingBARTLargeMNLI|Encoding: zero-shoot classification|Implemented|
|DeBERTaV3 Base|LLMEncodingDeBERTaV3Base|Encoding: zero-shoot classification|Implemented|
|DeBERTaV3 Base|LLMEncodingDeBERTaV3BaseMNLI|Encoding: zero-shoot classification|Implemented|
|DeBERTaV3 Large|LLMEncodingDeBERTaV3Large|Encoding: zero-shoot classification|Implemented|

**Note:** _NLP transformers._

### LVM (Large Vision Models)

|Source model|PyYel model|Task|Status|
|------------|-----------|----|------|
|ViT|LVMVisionTransformerClassification|Classification|TODO|

**Note:** _Computer vision transformers._

## Notes
See also [***PyYel Dev Ops***](https://github.com/PyYel/PyYel-DevOps) and [***PyYel Cloud Ops***](https://github.com/PyYel/PyYel-CloudOps) for development and deployment tools.