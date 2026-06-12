
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from safetensors import safe_open
from collections import OrderedDict

from typing import List, Optional, Tuple, Union

import torch
import transformers.utils.import_utils as transformers_import_utils

transformers_import_utils._sklearn_available = False

from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3Model, Qwen3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ex_omni.model.llava_her_arch import LlavaHerMetaForCausalLM
from ex_omni.model.speech_generator.generation import GenerationWithCTC

from ..speech_encoder.builder import build_speech_encoder
from ..speech_projector.builder import build_speech_projector
from ..speech_generator.builder import build_speech_generator as build_ctc_speech_generator
from ..speech_generator.builder import build_speech_generator as build_ar_speech_generator
from ..blendshape_generator.builder import build_blendshape_generator


class LlavaHerQwenConfig(Qwen3Config):
    model_type = "llava_her_qwen3"


class LlavaHerQwen3Model(Qwen3Model):
    config_class = LlavaHerQwenConfig
    def __init__(self, config: Qwen3Config):
        super(LlavaHerQwen3Model, self).__init__(config)

class LlavaHerQwen3ForCausalLM(Qwen3ForCausalLM, LlavaHerMetaForCausalLM):
    config_class = LlavaHerQwenConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def initialize_speech_modules(self):
        param = next(self.parameters())
        device = param.device
        dtype = param.dtype

        speech_encoder = build_speech_encoder(self.config)
        self.config.speech_encoder_type = "whisper"
        self.model.speech_encoder = speech_encoder.to(device=device, dtype=dtype)
        self.config.speech_encoder_hidden_size = self.model.speech_encoder.hidden_size

        self.model.speech_projector = build_speech_projector(self.config).to(
            device=device,
            dtype=dtype
        )

        if self.config.speech_generator_type == 'ar':
            self.model.speech_generator = build_ar_speech_generator(self.config).to(
                device=device,
                dtype=dtype
            )
        elif self.config.speech_generator_type == 'ctc':
            self.model.speech_generator = build_ctc_speech_generator(self.config).to(
                device=device,
                dtype=dtype
            )

    def initialize_blendshape_modules(self):
        param = next(self.parameters())
        device = param.device
        dtype = param.dtype

        self.model.blendshape_generator = build_blendshape_generator(self.config).to(
            device=device,
            dtype=dtype
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        重写 from_pretrained 方法，自动加载语音模块权重
        """
        # 首先使用父类的 from_pretrained 方法（此时已经包含了所有权重）
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # 初始化语音模块
        model.initialize_speech_modules()

        # 初始化blendshape模块
        model.initialize_blendshape_modules()
        
        # 从已加载的模型中提取语音模块权重（更高效）
        model.load_existing_weights(
            pretrained_model_name_or_path,
            load_speech_weights=True,
            load_blendshape_weights=True,
        )

        return model

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        streaming_unit_gen=False,
        faster_infer=False,
        text_tokens=False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if speech is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if faster_infer:
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            ), None
        else:
            outputs = GenerationWithCTC.generate(
                self,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict_in_generate=True,
                streaming_unit_gen=streaming_unit_gen,
                **kwargs
            )

            hidden_states = outputs['hidden_states']
            hidden_states = torch.cat([hidden_states[0][-1][:, -1:, :]] + [hidden_states[i][-1] for i in range(1, len(hidden_states))], dim=1)

            speech_generator = self.get_model().speech_generator
            target_device = next(speech_generator.parameters()).device
            target_dtype = next(speech_generator.parameters()).dtype

            if hidden_states.device != target_device or hidden_states.dtype != target_dtype:
                hidden_states = hidden_states.to(device=target_device, dtype=target_dtype)

            if outputs.sequences.device != target_device:
                outputs.sequences = outputs.sequences.to(device=target_device)

            speech_units_pred, speech_hidden_states = self.get_model().speech_generator.predict(hidden_states, outputs.sequences, return_hidden_states=True)
            blendshape_pred = self.get_model().blendshape_generator.predict(
                speech_hidden_states,
                torch.tensor(speech_units_pred, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            )

            speech_pred_str = ' '.join([str(x) for x in speech_units_pred])
            return outputs.sequences, speech_pred_str, blendshape_pred

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if speech is not None:
            inputs['speech'] = speech
            inputs['speech_lengths'] = speech_lengths
        return inputs


    def load_existing_weights(self, pretrained_model_path, load_speech_weights=True, load_blendshape_weights=True):
        print(f"Extracting speech weights from loaded model...")
        state_dict = self.load_safetensors_state_dict(pretrained_model_path)

        if load_speech_weights:
            self.load_speech_weights(state_dict)
        if load_blendshape_weights:
            self.load_blendshape_weights(state_dict)

    def load_speech_weights(self, state_dict):
        """
        从已加载的模型中提取语音模块权重并重新分配
        适用于模型已经通过 from_pretrained 加载的情况
        """
        # 提取语音模块权重
        speech_weights = self._extract_speech_weights_from_state_dict(state_dict)

        # 重新加载权重到对应模块（确保正确映射）
        self._load_speech_module_weights(speech_weights)
    
    def load_safetensors_state_dict(self, folder_path):
        """
        从safetensors文件加载状态字典
        
        Args:
            folder_path: 包含safetensors文件的文件夹路径
            
        Returns:
            OrderedDict: 加载的状态字典
        """
        state_dict = OrderedDict()

        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith(".safetensors"):
                file_path = os.path.join(folder_path, file_name)
                print(f"Loading {file_path}...")
                with safe_open(file_path, framework="pt") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        state_dict[key] = tensor

        return state_dict
    
    def _extract_speech_weights_from_state_dict(self, full_state_dict):
        """
        从完整的状态字典中提取语音模块权重
        
        Args:
            full_state_dict: 完整的模型状态字典
            
        Returns:
            提取的语音模块权重字典
        """
        speech_weights = {
            'speech_encoder': {},
            'speech_projector': {},
            'speech_generator': {}
        }
        
        # 定义各模块的键名前缀
        speech_module_prefixes = {
            'speech_encoder': ['model.model.speech_encoder.', 'model.speech_encoder.', 'speech_encoder.'],
            'speech_projector': ['model.model.speech_projector.','model.speech_projector.', 'speech_projector.'],
            'speech_generator': ['model.model.speech_generator.','model.speech_generator.', 'speech_generator.']
        }
        
        # 遍历完整状态字典，提取语音模块权重
        for key, value in full_state_dict.items():
            matched = False
            for module_name, prefixes in speech_module_prefixes.items():
                if matched:
                    break
                for prefix in prefixes:
                    if key.startswith(prefix):
                        # 移除前缀，得到模块内的键名
                        module_key = key[len(prefix):]
                        speech_weights[module_name][module_key] = value
                        print(f"✓ Extracted {module_name} weight: {key} -> {module_key}")
                        matched = True
                        break
        
        # 报告提取结果
        for module_name, weights in speech_weights.items():
            if weights:
                print(f"✓ Extracted {len(weights)} weights for {module_name}")
            else:
                print(f"✗ No weights found for {module_name}")
        
        return speech_weights
    
    def _load_speech_module_weights(self, loaded_weights):
        """
        将加载的权重应用到对应的模块
        
        Args:
            loaded_weights: 加载的权重字典
        """
        # 加载 speech_encoder
        if 'speech_encoder' in loaded_weights and hasattr(self.model, 'speech_encoder'):
            try:
                missing_keys, unexpected_keys = self.model.speech_encoder.load_state_dict(
                    loaded_weights['speech_encoder'], strict=False)
                if missing_keys:
                    print(f"Missing keys in speech_encoder: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys in speech_encoder: {unexpected_keys}")
                print("✓ speech_encoder weights loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load speech_encoder weights: {e}")
        
        # 加载 speech_projector
        if 'speech_projector' in loaded_weights and hasattr(self.model, 'speech_projector'):
            try:
                missing_keys, unexpected_keys = self.model.speech_projector.load_state_dict(
                    loaded_weights['speech_projector'], strict=False)
                if missing_keys:
                    print(f"Missing keys in speech_projector: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys in speech_projector: {unexpected_keys}")
                print("✓ speech_projector weights loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load speech_projector weights: {e}")
        
        if 'speech_generator' in loaded_weights and hasattr(self.model, 'speech_generator'):
            try:
                missing_keys, unexpected_keys = self.model.speech_generator.load_state_dict(
                    loaded_weights['speech_generator'], strict=False)
                if missing_keys:
                    print(f"Missing keys in speech_generator: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys in speech_generator: {unexpected_keys}")
                print("✓ speech_generator weights loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load speech_generator weights: {e}")

    def load_blendshape_weights(self, state_dict):
        blendshape_weights = {'blendshape_generator': {}}
        blendshape_module_prefixes = {
            'blendshape_generator': ['model.model.blendshape_generator.', 'model.blendshape_generator.', 'blendshape_generator.']
        }

        for key, value in state_dict.items():
            for module_name, prefixes in blendshape_module_prefixes.items():
                matched = False
                for prefix in prefixes:
                    if key.startswith(prefix):
                        module_key = key[len(prefix):]
                        blendshape_weights[module_name][module_key] = value
                        matched = True
                        break
                if matched:
                    break

        if 'blendshape_generator' in blendshape_weights and getattr(self.model, 'blendshape_generator', None) is not None:
            try:
                missing_keys, unexpected_keys = self.model.blendshape_generator.load_state_dict(
                    blendshape_weights['blendshape_generator'], strict=False)
                if missing_keys:
                    print(f"Missing keys in blendshape_generator: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys in blendshape_generator: {unexpected_keys}")
                print("✓ blendshape_generator weights loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load blendshape_generator weights: {e}")

AutoConfig.register("llava_her_qwen3", LlavaHerQwenConfig)
AutoModelForCausalLM.register(LlavaHerQwenConfig, LlavaHerQwen3ForCausalLM)
