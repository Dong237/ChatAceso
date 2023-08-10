import os
import sys

INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(INFERENCE_DIR, '..'))
sys.path.append("/cluster/home/Aceso")

import cmd
import conversation as convo
from augmentation.merkmanual import MerkManualDB
from tools.model import ChatModel
from tools.utils import parse_args, get_max_memory, MODULE_DIR



class OpenChatKitShell(cmd.Cmd):
    intro = "Welcome to Aceso.   Type /help or /? to list commands.\n"
    prompt = ">>> "

    def __init__(
            self, 
            gpu_id, 
            model_name_or_path, 
            cache_dir, 
            peft, 
            peft_weights, 
            max_tokens, 
            sample, 
            temperature, 
            top_k,
            repetition_penalty,
            retrieval, 
            max_memory, 
            retrieval_data_path,
            summarizer_device,
            ):
        super().__init__()
        self._gpu_id = gpu_id
        self._model_name_or_path = model_name_or_path
        self._cache_dir = cache_dir
        self._peft = peft
        self._peft_weights = peft_weights
        self._max_tokens = max_tokens
        self._sample = sample
        self._temperature = temperature
        self._top_k = top_k
        self._repetition_penalty = repetition_penalty
        self._retrieval = retrieval
        self._max_memory = max_memory
        self._retrieval_data_path = retrieval_data_path
        self._summarizer_device = summarizer_device
        
    def preloop(self):
        print(f"Loading {self._model_name_or_path} to cuda:{self._gpu_id}...")
        self._model = ChatModel(
            self._model_name_or_path, 
            self._cache_dir,
            self._peft, 
            self._peft_weights, 
            self._gpu_id, 
            self._max_memory
            )

        if self._retrieval:
            self._retriever = MerkManualDB(
                db_path=self._retrieval_data_path,
                device=self._summarizer_device
                )
            print(f"Augmentation Database Loaded")

        self._convo = convo.Conversation()

    def precmd(self, line):
        if line.startswith('/'):
            return line[1:]
        else:
            return 'say ' + line

    def do_say(self, arg):
        if self._retrieval:
            results = self._retriever.search(query=arg)
            if len(results) > 0:
                self._convo.push_context_turn(results)

        self._convo.push_human_turn(arg)

        inference_config = {
            'max_new_tokens': self._max_tokens,
            'do_sample': self._sample,
            'temperature': self._temperature,
            'top_k': self._top_k,
            "repetition_penalty": self._repetition_penalty
        }

        output = self._model.do_inference(
            prompt=self._convo._prompt,
            **inference_config
        )

        self._convo.push_model_response(output)
        print(output)
    
    def do_context_info(self, arg):
        print(self._convo._context)

    def do_raw_say(self, arg):
        output = self._model.do_inference(
            arg,
            self._max_tokens,
            self._sample,
            self._temperature,
            self._top_k
        )

        print(output)

    def do_raw_prompt(self, arg):
        print(self._convo.get_raw_prompt())

    def do_reset(self, arg):
        self._convo = convo.Conversation(
            self._model.human_id, self._model.bot_id)

    def do_hyperparameters(self, arg):
        print(
            f"Hyperparameters:\n"
            f"  max_tokens: {self._max_tokens}\n"
            f"  sample: {self._sample}\n"
            f"  temperature: {self._temperature}\n"
            f"  top_k: {self._top_k}"
        )

    def do_quit(self, arg):
        return True


def main():

    args = parse_args()
    max_memory = get_max_memory(args.gpu_vram, args.cpu_ram)

    OpenChatKitShell(
        args.gpu_id,
        args.model,
        args.cache_dir,
        args.peft,
        args.peft_weights,
        args.max_tokens,
        args.sample,
        args.temperature,
        args.top_k,
        args.repetition_penalty,
        args.use_retrieval,
        max_memory,
        args.retrieval_data_path,
        args.summarizer_device,
    ).cmdloop()


if __name__ == '__main__':
    main()