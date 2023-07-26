---
title: Llama 2 13b Chat
emoji: 🦙
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 3.37.0
app_file: app.py
pinned: false
license: other
suggested_hardware: a10g-small
duplicated_from: huggingface-projects/llama-2-13b-chat
---

# LLAMA v2 Models running Locally
Llama v2 was introduced in [this paper](https://arxiv.org/abs/2307.09288).

blog [llam2] (https://huggingface.co/blog/llama2)
quantization and QLoRa[QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

This Space demonstrates [Llama-2-13b-chat-hf](meta-llama/Llama-2-13b-chat-hf) from Meta. Please, check the original model card for details.

# prerequsite 
obtain access to Llama v2 models from Meta in HF

# generate a self-signed certificate
```
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes
```
# local setup
conda env create -f environment.yml
pip install -r requirements.txt
conda install -c anaconda cudatoolkit

# run
python app.py
