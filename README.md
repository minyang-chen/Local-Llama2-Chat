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

# Supervised Fine-tuning Trainer
https://huggingface.co/docs/trl/v0.4.7/en/sft_trainer

tutorial: https://medium.com/@ud.chandra/instruction-fine-tuning-llama-2-with-pefts-qlora-method-d6a801ebb19

dataset: https://huggingface.co/datasets/timdettmers/openassistant-guanaco

```
python scripts/sft_trainer.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name timdettmers/openassistant-guanaco \
    --load_in_4bit \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2

```
quick and dirty approach to load the model and do a sanity test.

```
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map=device_map, torch_dtype=torch.bfloat16)
text = "..."
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs["attention_mask"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```