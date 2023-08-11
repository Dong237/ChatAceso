# ChatAceso

## Getting Started

ChatAceso is a finetuned LLM based on LLaMA-2 model from Meta

## Installation

```bash
pip install -r requirements.txt
```

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

## Data

## Inference

**Command Line inference**

Command line inference is adapted from [OpenChatKit](https://github.com/togethercomputer/OpenChatKit/tree/main/inference). An example for starting the inference script can be found [here](https://github.com/Dong237/ChatAceso/tree/main/inference#inferencing)

**Web inference**

ChatAceso is also available on [hugginface space](https://huggingface.co/spaces/Dong237/ChatAceso).

## Evaluation

## License

## Acknowledgements

