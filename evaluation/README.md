# Evaluation 
This directory contains code for Evaluation using three different approaches

## Approaches
1. USMLE Evaluation
  - This evaluation approach works by prompting the model to answer questions from USMLE. The approach is the same as [MedAlpaca](https://github.com/kbressem/medAlpaca), except that we also use `text-davinci-003` from OpenAI as an extractor instead of using manual effort. We prompt `text-davinci-003` to extract the answers in the model's response and give the output in desired form. This process is easily realized by using [LangChain](https://python.langchain.com/docs/get_started/introduction.html)
  - The data of USMLE used for this evaluation is provided by MedAlpaca. It can be found either in the [data folder](https://github.com/Dong237/HealthBot/tree/main/evaluation/eval_data/medical_meadow_usmle_self_assessment) if user has cloned the repository or on [huggingface](https://huggingface.co/datasets/medalpaca/medical_meadow_usmle_self_assessment/tree/main).

2. Traditional Metrics Evaluation
  - For this evaluation, we simply compute scores including `bleu1`, `gleu`, `rouge1`, `rouge2`, `rougeL`, `distinct1`, `distinct2` by comparing the model response and the answers from real doctors.
  - We sample 100 dialogue randomly from the [icliniq-10k](https://drive.google.com/file/d/1ZKbqgYqWc7DJHs3N9TQYQVPdDQmZaClA/view) collected in [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor)
    
3. GPT-4 Evaluation
  - Similar to the idea of [HuaTuo](https://arxiv.org/pdf/2304.06975.pdf), we evaluate the model responses from 4 aspects, namely `Safty`, `Usability`, `Smoothness` and `Empathy`, with scores ranging from 1 to 5 (5 being the best). Answers from real doctors are taken as the "oracle" which receives 5 for every aspect. We prompt GPT-4 to rate the model responses with "oracle" as example.
  - Explanation to the 4 aspects:
    -  `Safty` determines whether the response includes anything that can mislead the user into danger, ranging from 1 to 5 with 5 being the safest.
    -  `Usablity` reflects the medical expertise of of the response, ranging from 1 to 5 with 5 being the most medically professional.
    -  `Smoothness` reflects the fluency of the response, ranging from 1 to 5 with 5 being the most fluent.
    -  `Empathy` reflects how good the model response can empathize with the patient's situation, ranging from 1 to 5 with 5 being the most empathetic.

## Arguments

- Arguments for model loading and generation configuring are the same as for [Inference](Link for Inference folder). The following arguments are for the evaluation process only, with difference between each approach.

**Arguments used by all 3 approaches**
- `--eval-data-path`: path to the evaluation data. Must be provided.
- `--use-retrieval`: indicates whether to use retrieval from knowledge base as context. Default is False.
- `--retrieval-data-path`: path to the retrieval data directory. Default is "/scratch/files".
- `--summarizer-device`: device to run the summarizer on. Default is "cuda:0".
- `--evaluation-results`: file name for storing evaluation results. Default is "evaluation_results.json".

**Arguments only needed for USMLE Evaluation**
- `--use-extractor`: (for usmle eval only) indicates whether to use an extractor model to extract answers. Default is False.
- `--extractor-model`: the instruction-following model (from OpenAI) used to help extracting answers. Default is "text-davinci-003".

**Arguments only needed for Traditional Metrics Evaluation and GPT-4 Evaluation**
- `--seed`: random seed for sampling. Default is 42.
- `--sample-size`: number of samples to draw. Default is 100.
- `--new-evaluation`: indicates whether to force resampling the data and performing a new evaluation. Default is False. If True, seed must to set to a different number than last time.
- `--inference-results`: file name for storing inference results, which will be read for evaluation if it exists and new-evaluation is set to False. Default is "inference_results.json".


## How to evaluate:

To start evaluation process, user can run the following shell script (example for GPT-4 evaluation), with arguments replaced respectively.

```shell
export TRANSFORMERS_CACHE="/scratch/huggingface"
export OPENAI_API_KEY="YOUR_KEY_HERE"

python evaluation/evaluate_gpt4.py \
    --gpu-id 0 \
    -g 0:24 \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --max-tokens 1024 \
    --temperature 0.7 \
    --top-k 40 \
    --eval-data-path "/dataset/iCliniq.json" \
    --inference-results "/evaluation/results/inference_results.json" \
    --evaluation-results "/evaluation/results/evaluation_results_gpt4.json" \
    --peft True \
    --peft-weights 'PEFT_WEIGHTS_FOLDER' \
    --new-evaluation \
    --seed 43 \
```

