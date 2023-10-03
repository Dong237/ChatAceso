import os
import fire
import torch
from tqdm import tqdm
from typing import List
from torch.optim import Adam
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from trl import (
    AutoModelForCausalLMWithValueHead, 
    PPOConfig, 
    PPOTrainer, 
    create_reference_model, 
    set_seed
)
from peft import PeftModel, LoraConfig
tqdm.pandas()


PROMPT_DICT = {
    "prompt_input_output": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:{output}"
        ),
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
    "prompt_output":(
        "{output}"
    )
        }

REMOVE_COLUMNS=["conv_id", "prompt", "utterance"]
INSTRUCTION = """\
You are a doctor named Aceso, interact with the person, answer their queries and be empathetic, helpful, and safe."""

def generate_prompt(data_point, prompt_type="input_output"):  
    if prompt_type == "input_output": # for training
        return PROMPT_DICT["prompt_input_output"].format_map(data_point)
    elif prompt_type == "input": # for evaluation
        return PROMPT_DICT["prompt_input"].format_map(data_point)
    else: # for evaluation'
        return PROMPT_DICT["prompt_output"].format_map(data_point)
    
def train(
    # model/data params
    base_model_name: str = "",
    reward_model_name: str = "bdotloh/distilbert-base-uncased-empathetic-dialogues-context",
    dataset_name: str = "Dong237/empathetic_dialogues_cleaned",
    cache_dir: str = "/scratch/huggingface/",
    output_dir: str = "./output",
    use_cache: bool = False,
    # training hyperparams
    load_in_4bit: bool = True,
    cutoff_len: int = 512,
    learning_rate=1.41e-5,
    ppo_epochs=1,
    gradient_accumulation_steps=1,
    mini_batch_size=1,
    batch_size=4,
    max_new_tokens: int = 1024,
    verbose: int = 1,
    # lora hyperparams
    lora_r: int = 2,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    peft_weights_dir: str = "./peft_weights",
    # logging params
    log_with: str = "wandb",      
    wandb_project: str = "trl-test",
    wandb_run_name: str = "run-1",
    ):


    ### PPO config
    tracker_kwargs = {
        "wandb_project": wandb_project,
        "run_name": wandb_run_name,
    }
    tracker_project_name = tracker_kwargs["wandb_project"]

    config = PPOConfig(
        model_name=base_model_name,    
        learning_rate=learning_rate,
        ppo_epochs=ppo_epochs,
        mini_batch_size=mini_batch_size,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        remove_unused_columns=False,
        log_with=log_with,
        tracker_kwargs=tracker_kwargs,
        tracker_project_name=tracker_project_name,
    )

    ### device map
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 0))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    ### Model, Optimizer, Tokenizer
    set_seed(config.seed)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype="bfloat16", 
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        )

    if os.path.exists(peft_weights_dir):
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = PeftModel.from_pretrained(
            model, 
            peft_weights_dir,
            lora_config=lora_config,
            torch_dtype=torch.bfloat16, 
            device_map="auto",                                       
            is_trainable=True
        ) 
        model.print_trainable_parameters()  
        print("="*50)
        print("Loaded LoRA model from {}".format(peft_weights_dir))

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        is_trainable=True,
        quantization_config=bnb_config,
        device_map=device_map,
        )
    model.config.use_cache = use_cache
    ref_model = create_reference_model(model)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side="left",
        )
    tokenizer.pad_token = tokenizer.eos_token

    ### Dataset
    def restructure_data_for_training(datapoint):
        datapoint["utterance"]=datapoint["utterance"].replace("_comma_", ",")
        datapoint["utterance"]=datapoint["utterance"].split(" <SEP> ")[:2]
        datapoint["instruction"] = INSTRUCTION
        datapoint["input"] = datapoint["utterance"][0]
        datapoint["output"] = datapoint["utterance"][1]
        return datapoint

    def tokenize(datapoint):
        prompt = generate_prompt(datapoint, prompt_type="input")
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            return_token_type_ids=False
        )
        datapoint["input_ids"] = result["input_ids"]
        datapoint["attention_mask"] = result["attention_mask"]
        datapoint["query"] = tokenizer.decode(result["input_ids"])
        return datapoint

    def build_dataset(dataset_name):
        if dataset_name.endswith(".json"):  
            dataset = load_dataset("json", data_files=dataset_name)
        else:
            dataset = load_dataset(dataset_name)
        dataset = dataset.map(restructure_data_for_training, remove_columns=REMOVE_COLUMNS)
        dataset = dataset.map(tokenize)
        dataset.set_format(type="torch")
        return dataset
    
    def collator(data):
        keys_to_collect = ['context', 'input_ids', 'query']
        return dict((key, [d[key] for d in data if key in d]) for key in keys_to_collect)

    dataset = build_dataset(dataset_name=dataset_name)

    ### PPO Trainer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset["train"],
        data_collator=collator,
        optimizer=optimizer,
    )

    ## Reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        ).to(ppo_trainer.accelerator.device)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

    def compute_reward(text_batch: list, context_batch: list):
        if len(text_batch) != len(context_batch):
            raise ValueError("text_batch and context_batch must have the same length")
        input_ids = reward_tokenizer(
            text_batch, 
            padding=True,
            truncation=True,
            return_tensors="pt")["input_ids"].to(reward_model.device)
        logits = reward_model(input_ids=input_ids).logits
        context_ids = [
            reward_model.config.label2id[context] 
            for context in context_batch
            ]
        reward_tensor = [
            logits[i][context_ids[i]].detach() 
            for i in range(len(context_ids))]
        return reward_tensor

    ## PPO training

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": max_new_tokens,
    }

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):

        query_tensors = batch["input_ids"]
        response_tensors = []
        contexts = batch["context"]

        for query in tqdm(query_tensors, desc="generating responses from batch"):
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[len(query):])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]  
        rewards = compute_reward(query_response_pairs, contexts)

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if verbose == 1:
            print(f'objective/kl: {stats["objective/kl"]}')
            print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
            print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
            print('-'.join('' for _ in range(100)))
        
        if step % 100 == 0:
            if ppo_trainer.accelerator.is_main_process:
                ppo_trainer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)