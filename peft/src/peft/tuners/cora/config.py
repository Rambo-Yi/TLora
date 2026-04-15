# Copyright 2023-present the HuggingFace Inc. team.
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
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class VeraConfig(PeftConfig):
    r: int = field(default=256)

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
    )
    projection_prng_key: int = field(
        default=0,
        metadata={
            "help": (
                "Vera PRNG init key. Used for initialising vera_A and vera_B for new models or when loading a "
                "checkpoint that did not include these projections."
            )
        },
    )
    save_projection: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to save the vera_A / vera_B projections in the state dict alongside per layer lambda_b / "
                "lambda_d weights. This will increase the size of the checkpoint, but guarantee that we can reload "
                "the checkpoint on all system configurations."
            )
        },
    )
    vera_dropout: float = field(default=0.0)
    d_initial: float = field(default=0.1, metadata={"help": "Initial init value for d vector."})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "如果要替换的层存储的权重类似于（fan_in，fan_out），则将其设置为True"},
    )
    bias: str = field(default="none")
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "该参数允许用户指定除Vera 层之外，其他哪些模块（层）应该是可训练的，"
                "并且会在最终模型检查点中保存。这通常用于如序列分类、标记分类等任务中，最后一层（如分类器层）需要训练并保存。"
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the Vera layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "指定需要应用 Vera 转换的层索引。如果传递一个整数，则表示仅应用 Vera 转换到该层。如果传递一个整数列表，则仅应用 Vera 转换到这些指定的层"
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer "
                "pattern is not in the common layers pattern. This should target the `nn.ModuleList` of the "
                "model, which is often called `'layers'` or `'h'`."
            )
        },
    )

    '''
    在 VeraConfig 实例化后执行一些初始化检查和设置。
    例如，它会检查 layers_pattern 和 layers_to_transform 是否正确搭配使用，
    同时如果 save_projection 设置为 False，则会给出警告，提醒用户这些投影矩阵将不会被保存。
    '''
    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.VERA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
        if not self.save_projection:
            warnings.warn(
                "Specified to not save vera_A and vera_B within the state dictionary, instead they will be restored "
                "using the PRNG key store in `config.projection_prng_key`. Consider setting `config.save_projection` "
                "to `True` to guarantee restoring the checkpoint correctly on all system configurations."
            )
