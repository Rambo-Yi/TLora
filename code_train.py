import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import logging
import os
import fire
import torch
import torch.distributed
import transformers
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
import numpy as np
import wandb
from peft.tuners.lora.tlora import preprocess_tlora
from peft import LoraConfig, TaskType, get_peft_model

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from tqdm import tqdm


def train(
        # 模型参数
        base_model: str = "Llama-2-7b-hf",  # the only required argument
        data_path: str = "CodeFeedback-Filtered-Instruction.json",
        data_split: str = "train[:100000]",
        dataset_field: list[str] = ["query", "answer"],
        output_dir: str = "./results/code/",
        use_cache: bool = True,
        # 训练超参数
        batch_size: int = 128,
        micro_batch_size: int = 8,
        num_epochs: int = 1,
        learning_rate: float = 2e-5,
        cutoff_len: int = 512,
        # LoRA超参数
        lora_r: int = 128,
        lora_alpha: int = 128,
        lora_dropout: float = 0,
        lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"use_cache: {use_cache}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n")
    IGNORE_INDEX = -100
    PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 确保所有 GPU 设定相同的种子
        torch.backends.cudnn.deterministic = True  # 保证每次结果一致
        torch.backends.cudnn.benchmark = False  # 关闭自动优化以保证可复现性
        torch.use_deterministic_algorithms(True)

    set_seed(42)  # 在训练前设定随机种子

    gradient_accumulation_steps = batch_size // micro_batch_size

    def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length, truncation=True, ) for text in strings]
        input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(
            sources: Sequence[str],
            targets: Sequence[str],
            tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    '''
        这个类的作用是 用于批量（Batch）整理数据（Collate），
        确保 input_ids 和 labels 具有相同的长度，并生成 attention_mask，适用于 监督微调（Supervised Fine-tuning, SFT）
    '''

    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""
        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            input_ids = [torch.tensor(x) for x in input_ids]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = [torch.tensor(x) for x in labels]
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )

        '''
        该函数用于 将文本数据转换为模型训练所需的 Token 格式
        '''

    def train_tokenize_function(examples, tokenizer, query, response):
        sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
        targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
        data_dict = preprocess(sources, targets, tokenizer)
        return data_dict

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        model_max_length=cutoff_len,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        use_cache=use_cache,
    )
    if not use_cache:
        model.gradient_checkpointing_enable()

    dataset = load_dataset("json", data_files=data_path,
                           split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(256))

    train_dataset = dataset.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": dataset_field[0],
                   "response": dataset_field[1]}
    )

    def run_model():
        for batch in tqdm(train_dataset):
            model.zero_grad()
            input_ids = torch.tensor(batch["input_ids"]).to(model.device).unsqueeze(0)
            labels = torch.tensor(batch["labels"]).to(model.device).unsqueeze(0)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_target_modules,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_lora_weights="tlora",
    )
    preprocess_tlora(model, peft_config, run_model=run_model)

    model = get_peft_model(model, peft_config)
    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)

    model.print_trainable_parameters()

    raw_train_datasets = load_dataset("json", data_files=data_path, split=data_split)

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": dataset_field[0],
                   "response": dataset_field[1]}
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    wandb.init(
        mode="offline",  # 或者 "online" 取决于你是否要上传数据
        project="commonsense",  # 指定项目名称
        name=output_dir,  # 这里是你要设置的 WandB 运行名称
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.03,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=1000,
            output_dir=output_dir,
            save_total_limit=1,
            label_names=["labels"],
            lr_scheduler_type="cosine",
        ),
        train_dataset=train_dataset,

        data_collator=data_collator)

    trainer.train()
    trainer.save_state()
    model.peft_config["default"].init_lora_weights = True
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(train)