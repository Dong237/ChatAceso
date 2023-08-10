import sys
from typing import Any, List, Optional
from tools.utils import get_max_memory, clean_response

sys.path.append("/cluster/home/Aceso")

import torch
from peft import PeftModel
from pydantic import Extra
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig, 
    StoppingCriteria, 
    StoppingCriteriaList
)
from accelerate import infer_auto_device_map, init_empty_weights
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


class StopWordsCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_words):
        self._tokenizer = tokenizer
        self._stop_words = stop_words
        self._partial_result = ''
        self._stream_buffer = ''

    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
            ) -> bool:
        text = self._tokenizer.decode(input_ids[0, -1])
        self._partial_result += text
        for stop_word in self._stop_words:
            if stop_word in self._partial_result:
                return True
        return False


class ChatModel:
    eos_tag = "[/INST]"

    def __init__(
            self, 
            model_name, 
            cache_dir, 
            peft, 
            peft_weights, 
            gpu_id, 
            max_memory
            ):
        device = torch.device('cuda', gpu_id)

        # recommended default for devices with > 40 GB VRAM
        # load model onto one device
        if max_memory is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir, 
                torch_dtype=torch.float16, 
                device_map="auto"
                ) 
            self._model.to(device)
        # load the model with the given max_memory config (for devices with insufficient VRAM or multi-gpu)
        else:
            config = AutoConfig.from_pretrained(model_name,cache_dir=cache_dir)
            # load empty weights
            with init_empty_weights():
                model_from_conf = AutoModelForCausalLM.from_config(config)

            model_from_conf.tie_weights()

            # create a device_map from max_memory
            device_map = infer_auto_device_map(
                model_from_conf,
                max_memory=max_memory,
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype="float16"
            )
            # load the model with the above device_map
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                device_map=device_map,
                offload_folder="offload",  # optional offload-to-disk overflow directory (auto-created)
                offload_state_dict=True,
                torch_dtype=torch.float16
            )
        if peft:
            print("=====================================================")
            print("Loading PEFT model")
            self._model = PeftModel.from_pretrained(self._model, peft_weights) 
        else:
            print("=====================================================")
            print("Loading Base model")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
            )

    def do_inference(
            self, 
            prompt, 
            max_new_tokens, 
            do_sample, 
            temperature, 
            top_k,
            repetition_penalty,
            ):
        if prompt.endswith(self.eos_tag):
            prompt_formatted = prompt
        else:
            prompt_formatted = f"{prompt} {self.eos_tag}"

        inputs = self._tokenizer(
            prompt_formatted,
            return_token_type_ids=False, 
            return_tensors='pt',
            ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, # do_sample
            temperature=temperature,
            top_k=top_k,
            pad_token_id=self._tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
        )

        output_raw = self._tokenizer.batch_decode(outputs)[0]
        output = clean_response(output_raw)
        output = output.split(self.eos_tag)[-1].strip()
        return output


# this class is only for the purpose of using LangChain
class ChatLLM(LLM, extra=Extra.allow):
    _model: Any 
    _inference_config: Any
    human_tag: Any
    bot_tag: Any

    def __init__(
            self, 
            model_name_or_path, 
            cache_dir,
            peft, 
            peft_weights, 
            gpu_id, 
            gpu_vram, 
            cpu_ram, 
            inference_config
            ):
        super().__init__()
        max_memory=get_max_memory(gpu_vram, cpu_ram)
        self._model = ChatModel(
            model_name=model_name_or_path, 
            cache_dir=cache_dir,
            peft=peft, 
            peft_weights=peft_weights, 
            gpu_id=gpu_id, 
            max_memory=max_memory,
            )
        self._inference_config = inference_config

    @property
    def _llm_type(self) -> str:
        return type(self._model)

    def _call(
            self, 
            prompt,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None
            ):
        
        output = self._model.do_inference(
            prompt=prompt, 
            **self._inference_config,
            )
        return output