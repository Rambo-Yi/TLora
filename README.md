
# TLoRA: Task-aware Low Rank Adaptation of Large Language Models (ACL2026 Main)
The code for  TLoRA: Task-aware Low Rank Adaptation of Large Language Models (ACL2026 Main)

Low-Rank Adaptation (LoRA) has become a widely adopted parameter-efficient fine-tuning method for large language models, with its effectiveness largely influenced by the allocation of ranks and scaling factors, 
as well as initialization. Existing LoRA variants typically address only one of these factors, often at the cost of increased training complexity or reduced practical efficiency. In this work,
we present Task-aware Low-Rank Adaptation (TLoRA), a unified framework that jointly optimizes initialization and resource allocation at the outset of training. TLoRA introduces a data-driven initialization strategy that aligns the LoRA 
 A matrix with task-relevant subspaces by performing singular value decomposition on the product of pre-trained weights and input activation covariance. After this, the 
 matrix A is frozen, and only the 
 matrix B is trained. Furthermore, TLoRA employs a sensitivity-based importance metric to adaptively allocate ranks and scaling factors across layers under a fixed parameter budget. 
 We conduct extensive experiments that demonstrate TLoRA consistently performs excellently across various tasks, including natural language understanding, commonsense reasoning, 
 math reasoning, code generation, and chat generation, while significantly reducing the number of trainable parameters.


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

| Method| Trainable Params | GSM8K | MATH | HumanEval | MBPP | MT-Bench |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FULL** | 6738M | 52.31 | 8.08 | 23.20 | 38.60 | 4.75 |
| **LoRA** | 319.81M | 44.80 | 6.18 | 20.70 | 35.70 | 4.76 |
| **AdaLoRA** | 319.84M | 43.20 | 5.74 | 20.70 | 36.00 | 4.50 |
| **LoRA+** | 319.81M | 48.67 | 6.92 | 22.60 | 35.40 | 4.69 |
| **DoRA** | 321.17M | 45.10 | 5.96 | 20.70 | 36.00 | 4.70 |
| **OLoRA** | 319.81M | 52.38 | 8.22 | 22.00 | 38.90 | 4.99 |
| **PISSA** | 319.81M | 53.44 | 7.40 | 22.60 | 38.60 | 5.00 |
| **LoRA-GA** | 319.81M | **58.15** | 8.66 | 23.20 | 38.60 | 4.97 |
| **CorDA** | 319.81M | 54.43 | 8.70 | 21.30 | 39.40 | 5.09 |
| **TLoRA** | **171.71M** | 56.34 | **9.08** | **23.50** | **40.20** | **5.17** |

---