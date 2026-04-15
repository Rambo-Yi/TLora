import os
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from peft import (
    get_peft_model,
    LoraConfig
)
from peft.tuners.lora.tlora import preprocess_tlora
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np


def train(
    # 模型参数
    base_model: str = "Llama-2-7b-hf",
    data_path: str = "commonsense_170k.json",
    output_dir: str = "./",
    use_cache: bool = True,
    # 训练超参数
    batch_size: int = 32,
    micro_batch_size: int =8,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    cutoff_len: int = 512,
    val_set_size: int = 0,
    train_on_inputs: bool = False ,  # False则会掩盖输入部分
    group_by_length: bool = False,  # 会训练得更快，但是可能会训练效果不佳
    seed = 42,
    # LoRA超参数
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0,
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj","o_proj","gate_proj"],
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"use_cache: {use_cache}\n"
            f"batch_size: {batch_size}\n"
            f"weight_decay: {weight_decay}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"

        )

    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(seed)

    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        use_cache=use_cache,
        device_map=device_map,
    )

    if not use_cache:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = (
        0  # 使pad与eos有所区分
    )
    tokenizer.padding_side = "left"  # 方便批量推理

    dataset = load_dataset("json", data_files="commonsense_170k.json", split="train").shuffle(seed=42)
    dataset = dataset.select(range(256))

    def generate_prompt(data_point):
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                       ### Instruction:
                       {data_point["instruction"]}

                       ### Input:
                       {data_point["input"]}
                       

                       ### Response:
                       {data_point["output"]}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                       ### Instruction:
                       {data_point["instruction"]}

                       ### Response:
                       {data_point["output"]}"""

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]
        return tokenized_full_prompt


    def run_model():
        prompt = (dataset.shuffle().map(generate_and_tokenize_prompt,num_proc=64))
        for batch in tqdm(prompt):
            model.zero_grad()
            input_ids =  torch.tensor(batch["input_ids"]).to(model.device).unsqueeze(0)
            labels = torch.tensor(batch["labels"]).to(model.device).unsqueeze(0)
            outputs = model(input_ids=input_ids,labels=labels)
            loss = outputs.loss
            loss.backward()

    peft_config = LoraConfig(
        init_lora_weights='tlora',
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    preprocess_tlora(model, peft_config, run_model=run_model)
    model = get_peft_model(model=model, peft_config=peft_config)
    model.print_trainable_parameters()


    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt,num_proc=128)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt,num_proc=128)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt,num_proc=128)
        val_data = None



    '''训练模型'''
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=SFTConfig(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.03,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=0,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 else False,
            group_by_length=group_by_length,
            label_names=["labels"],
            lr_scheduler_type="linear",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer.train()

    model.peft_config["default"].init_lora_weights = True

    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)