# Evaluation
This directory contains code for Evaluation on USMLE. Date can be found in [data folder](https://github.com/Dong237/HealthBot/tree/main/evaluation/eval_data/medical_meadow_usmle_self_assessment) or downloaded from [MedAlpaca](https://huggingface.co/datasets/medalpaca/medical_meadow_usmle_self_assessment/tree/main).

Example of an bash script for Evaluation can be found [here]([https://github.com/Dong237/HealthBot/blob/main/inference.sh](https://github.com/Dong237/HealthBot/blob/main/evaluate.sh))

## Arguments
- `--model`: name/path of the model. Default = `../huggingface_models/GPT-NeoXT-Chat-Base-20B`
- `--peft`: indicates whether to use the model fine-tuned with PEFT, default is False
- `--peft-weights`: name/path of PEFT weights, only required when peft is used
- `--gpu-id`: Primary GPU device to load inputs onto for inference. Default: `0`
- `--max-tokens`: the maximum number of tokens to generate. Default: `128`
- `--sample`: indicates whether to sample. Default: `True`
- `--temperature`: temperature for the LM. Default: `0.6`
- `--top-k`: top-k for the LM. Default: `40`
- `--retrieval`: augment queries with context from the retrieval index. Default `False`
- `-g` `--gpu-vram`: GPU ID and VRAM to allocate to loading the model, separated by a `:` in the format `ID:RAM` where ID is the CUDA ID and RAM is in GiB. `gpu-id` must be present in this list to avoid errors. Accepts multiple values, for example, `-g ID_0:RAM_0 ID_1:RAM_1 ID_N:RAM_N`
- `-r` `--cpu-ram`: CPU RAM overflow allocation for loading the model. Optional, and only used if the model does not fit onto the GPUs given.
- `--data-dir`: path to the evaluation data (USMLE self assessemnt) directory
- `--use-extractor`: whether to use an extractor model (from OpenAI) to extract answers from model response
- `--evaluation-results`: name of the json file to store final answers and accuracy results

## Evaluation Process
- Requirements on GPUs and instructions on how to allocate are the same as [inference](https://github.com/Dong237/HealthBot/tree/main/inference) 

- The evaluation uses [LangChain](https://python.langchain.com/docs/get_started/introduction.html)
  
  *picture demo to be added...*
