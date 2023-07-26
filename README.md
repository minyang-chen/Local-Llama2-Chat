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

# prompt Llama 2

The prompt template for the first turn looks like this:
```
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST]

```
full example
```
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden 😱 What should I do? [/INST]

```

As the conversation progresses, all the interactions between the human and the "bot" are appended to the previous prompt, enclosed between [INST] delimiters. The template used during multi-turn conversations follows this structure (🎩 h/t Arthur Zucker for some final clarifications):
```
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
```

The model is stateless and does not "remember" previous fragments of the conversation, we must always supply it with all the context so the conversation can continue. This is the reason why context length is a very important parameter to maximize, as it allows for longer conversations and larger amounts of information to be used. 
