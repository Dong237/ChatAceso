# ChatAceso

## Getting Started

ChatAceso is a finetuned LLM based on LLaMA-2 model from Meta

## Installation

pip install -r requirements.txt


## Finetuning

The finetuning process uses LoRA and the script is adapted based on [alpaca-lora](https://github.com/tloen/alpaca-lora). Users can run the following example for distributed training.

```shell
torchrun --nproc_per_node=4 --master_port=8888 train_lora.py \
  --base_model "meta-llama/Llama-2-7b-chat-hf" \
  --data_path '/Aceso-110k.json' \
  --output_dir './output' \
  --use_cache False \
  --batch_size 128 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 1e-4 \
  --cutoff_len 512 \
  --val_set_size 5000 \
  --save_strategy "steps" \
  --eval_step 200 \
  --save_step 200 \
  --logging_steps 20 \
  --report_to "wandb" \
  --wandb_project "project" \
  --wandb_run_name "run_1" \
```

## Inference

### Command Line inference
The script is adapted from OpenChatKit

Start the bot by calling `bot.py` from the root for the repo.

```shell
python inference/cli/bot.py \
    --gpu-id 0 \
    -g 0:24 \
    --model 'meta-llama/Llama-2-7b-chat-hf' \
    --sample True \
    --temperature 0.7 \
    --top-k 50 \
    --max-tokens 1024 \
    --repetition-penalty 1.1 \
    --peft True \
    --peft-weights 'peft_weights' \
```

This example will download the model checkpoint from huggingface and the peft weights from the specified folder. Modle will be loaded onto "cuda:0" in this case.

Loading the model can take some time, but once it's loaded, you are greeted with a prompt. Say hello.

```shell
$ python inference/cli/bot.py 
Welcome to Aceso.   Type /help or /? to list commands.

>>> Hello.
Hello human.

>>> 
```

Enter additional queries at the prompt, and the model replies. Under the covers, the shell is forming a prompt with all previous queries and passes that to the model to generate more text.

The shell also supports additional commands to inspect hyperparamters, the full prompt, and more. Commands are prefixed with a `/`.


### Web inference

ChatAceso is also available on [hugginface space](https://huggingface.co/spaces/Dong237/ChatAceso).

