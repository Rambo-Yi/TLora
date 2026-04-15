# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from typing import Optional

from transformers import PreTrainedModel

from .auto import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from .config import PeftConfig
from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_PREFIX_MAPPING
from .mixed_model import PeftMixedModel
from .peft_model import PeftModel
from .tuners.tuners_utils import BaseTuner, BaseTunerLayer
from .utils import _prepare_prompt_learning_config


def get_peft_model(
    model: PreTrainedModel,
    peft_config: PeftConfig,
    adapter_name: str = "default",
    mixed: bool = False,
    autocast_adapter_dtype: bool = True,
    revision: Optional[str] = None,
    low_cpu_mem_usage: bool = False,
) -> PeftModel | PeftMixedModel:
    """
    Returns a Peft model object from a model and a config, where the model will be modified in-place.

    Args:
        model：这是你提供的预训练模型，类型是 PreTrainedModel，通常是 Hugging Face 的模型。
        peft_config：配置对象，包含了微调模型时所需的一些配置。
        adapter_name：一个字符串，指定要使用的适配器的名称，默认值是 "default"。
        mixed：布尔值，表示是否允许不同类型的适配器混合使用。
        autocast_adapter_dtype：布尔值，是否对适配器的权重进行自动类型转换，通常用于处理精度问题。
        revision：字符串，表示模型的版本或修订号。
        low_cpu_mem_usage：布尔值，控制是否以低 CPU 内存的方式加载模型。
    """
    # 第一步：获取模型配置
    model_config = BaseTuner.get_model_config(model)
    #第二步：处理 peft_config 中的 base_model_name_or_path 属性
    old_name = peft_config.base_model_name_or_path
    new_name = model.__dict__.get("name_or_path", None)
    peft_config.base_model_name_or_path = new_name

    # Especially in notebook environments there could be a case that a user wants to experiment with different
    # configuration values. However, it is likely that there won't be any changes for new configs on an already
    # initialized PEFT model. The best we can do is warn the user about it.
    if any(isinstance(module, BaseTunerLayer) for module in model.modules()):
        warnings.warn(
            "You are trying to modify a model with PEFT for a second time. If you want to reload the model with a "
            "different config, make sure to call `.unload()` before."
        )

    if (old_name is not None) and (old_name != new_name):
        warnings.warn(
            f"The PEFT config's `base_model_name_or_path` was renamed from '{old_name}' to '{new_name}'. "
            "Please ensure that the correct base model is loaded when loading this checkpoint."
        )

    if revision is not None:
        if peft_config.revision is not None and peft_config.revision != revision:
            warnings.warn(
                f"peft config has already set base model revision to {peft_config.revision}, overwriting with revision {revision}"
            )
        peft_config.revision = revision

    if (
        (isinstance(peft_config, PEFT_TYPE_TO_CONFIG_MAPPING["LORA"]))
        and (peft_config.init_lora_weights == "eva")
        and not low_cpu_mem_usage
    ):
        warnings.warn(
            "lora with eva initialization used with low_cpu_mem_usage=False. "
            "Setting low_cpu_mem_usage=True can improve the maximum batch size possible for eva initialization."
        )

    prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(peft_config.peft_type)
    if prefix and adapter_name in prefix:
        warnings.warn(
            f"Adapter name {adapter_name} should not be contained in the prefix {prefix}."
            "This may lead to reinitialization of the adapter weights during loading."
        )

    if mixed:
        # note: PeftMixedModel does not support autocast_adapter_dtype, so don't pass it
        return PeftMixedModel(model, peft_config, adapter_name=adapter_name)

    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not peft_config.is_prompt_learning:
        return PeftModel(
            model,
            peft_config,
            adapter_name=adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    if peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](
        model,
        peft_config,
        adapter_name=adapter_name,
        autocast_adapter_dtype=autocast_adapter_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
