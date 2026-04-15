
# TLoRA:Task-aware Low Rank Adaptation of Large Language Models

We introduce Task-aware Low-Rank Adaptation (TLoRA), a novel parameter-efficient fine-tuning (PEFT) method that enhances
LoRA by incorporating task-aware initialization and adaptive assignment of rank and scaling factors. Extensive experimen
ts demonstrate that TLoRA consistently outperforms LoRA and its variants across a wide range of tasks, including natural
language understanding, commonsense reasoning, mathematical reasoning, code generation, and dialogue generation. Below, 
we provide a code example based on a large-scale model experiment:


## Requirements

Create a conda environment and install dependencies:

```setup
conda activate -n tlora python=3.10
conda activate tlora

pip install -r requirements.txt
pip install -e peft
```

## Datasets and Model
Install Llama-2-7B from huggingface
Install datasets (WizardLM, MetaMathQA, and CodeFeedback-Filtered-Instruction) from huggingface

## Reproduce Llama-2-7b results

To train the model in the paper, run this command:

For commonsense task,
```
deepspeed --include=localhost:$gpu_number commonsense_train.py --deepspeed config/ds_config_zero2_no_offload.json
```
For math task,
```
deepspeed --include=localhost:$gpu_number math_train.py --deepspeed config/ds_config_zero2_no_offload.json
```
For code task,
```
deepspeed --include=localhost:$gpu_number code_train.py --deepspeed config/ds_config_zero2_no_offload.json
```
For chat task,
```
deepspeed --include=localhost:$gpu_number chat_train.py --deepspeed config/ds_config_zero2_no_offload.json
```

## Evaluation

For math task,

```eval
python math/eval_gsm8k.py --model $model_path
python math/eval_math.py --model $model_path
```

For code task, we generate results with script below and evaluate its PASS@1 using EvalPlus,
```eval
python gen_vllm --model $model_path
python math/eval_math.py
```

For chat task, we use FastChat to generation and evaluate with GPT-4


## Results

Our model achieves the following performance on :

| **Method** | **GSM8K** | **MATH** | **HumanEval** | **MBPP** | **MT-Bench** |
|------------|-----------|----------|----------------|----------|--------------|
| LoRA       | 44.80     | 6.18     | 20.70          | 35.70    | 4.89         |
| DoRA       | 45.10     | 5.96     | 20.70          | 36.00    | 4.75         |
| PiSSA      | 53.44     | 7.40     | 22.60          | 38.60    | 5.02         |
| **TLoRA**  | **57.01** | **8.78** | **23.20**      | **40.70**| **5.86**     |




