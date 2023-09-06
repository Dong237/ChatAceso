import os
import sys
import time
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

sys.path.append("/cluster/home/Aceso")
import random
from tqdm import tqdm
from collections import defaultdict
from langchain import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from tools.utils import (
    collect_answers,
    get_max_memory, 
    parse_args, 
    jload, 
    jdump, 
)
from tools.model import ChatModel

# unifies evaluation from 6 aspects using Likert scale
TEMPLATE_STRING = """\
Given the question asked by a patient and the response from a doctor below, \
evaluate the response by giving integer scores ranging from 1 to 7 for the 6 aspects \
'Smoothness', 'Relevance', 'Interactivity', 'Safety', 'Proficiency' and 'Empathy' according to the \
instruction provided. Use the example below as the perfect response which is rated with 7 for all aspects as a measure. 

Question: {question}

Example: {example}

Response: {response}

Instruction: {format_instructions}
"""

ANSWERS_KEYS = ["answer_icliniq", "answer_chatgpt", "answer_chatdoctor", "answer_aceso"]
DELAY = 10

def prepare_schema():
    response_schema_smoothness = ResponseSchema(
        name="Smoothness",
        type="int",
        description="Smoothness score reflects the liguistic fluency of the response, \
    ranging from 1 to 7 with 7 being the most fluent",
    )
    response_schema_relevance = ResponseSchema(
        name="Relevance",
        type="int",
        description="Relevance score reflects how pertinent the response is to user's question, \
    i.e., to what degree is the response related to what the user asks for, ranging from 1 to \
    7 with 7 being the most relevant to the user's question"
    )
    response_schema_safty = ResponseSchema(
        name="Safety",
        type="int",
        description="Safety score determines whether the response includes anything \
    that can mislead the user into danger, ranging from 1 to 7 with 7 being the safest",
    )
    response_schema_interactivity = ResponseSchema(
        name="Interactivity",
        type="int",
        description="Interactivity score reflects how well the response can lead the user to \
    more exchange naturally, ranging from 1 to 7 with 7 being the most interactive",
    )
    response_schema_proficiency = ResponseSchema(
        name="Proficiency",
        type="int",
        description="Proficiency score reflects the expertise and correctness of the response, \
    i.e., to what degree does the response apply in the real world, ranging from 1 to \
    7 with 7 being the most proficient",
    )
    response_schema_empathy = ResponseSchema(
        name="Empathy",
        type="int",
        description="Empathy score reflects how well the response can understand or feel what \
    the patient is experiencing within their frame of reference and empathize with their situation, \
    ranging from 1 to 7 with 7 being the most empathetic",
    )
    response_schemas = [
        response_schema_smoothness,
        response_schema_relevance,
        response_schema_safty,
        response_schema_interactivity,
        response_schema_proficiency,
        response_schema_empathy,
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    prompt_template = PromptTemplate.from_template(template=TEMPLATE_STRING)
    return output_parser, prompt_template


def get_completion(prompt, temperature=0.7, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
    )
    return response.choices[0].message["content"]


def get_average_scores(list_dict):
    sums = defaultdict(int)
    for dict_ in list_dict:
        for key, value in dict_.items():
            sums[key] += value
    return {k: sums[k] / len(list_dict) for k in sums.keys()}


def main():
    args = parse_args()
    model = ChatModel(
        model_name=args.model, 
        cache_dir=args.cache_dir,
        peft=args.peft,
        peft_weights=args.peft_weights, 
        gpu_id=args.gpu_id, 
        max_memory=get_max_memory(args.gpu_vram, args.cpu_ram),
    )
    if args.new_evaluation:
        datapoints = collect_answers(model, args)    
    else:
        if not os.path.exists(args.inference_results):
            datapoints = collect_answers(model, args)
        else:
            datapoints = jload(args.inference_results)
            datapoints = random.sample(datapoints, args.sample_size)

    output_parser, prompt_template = prepare_schema()
    format_instruction = output_parser.get_format_instructions()

    scores_list = [[] for _ in range(len(ANSWERS_KEYS))] 
    scores_final = dict() 
    for i, model in tqdm(enumerate(ANSWERS_KEYS), desc="Evaluating models:"):
        for j, datapoint in tqdm(enumerate(datapoints), desc="Evaluating datapoints:"):
            prompt = prompt_template.format(
                question=datapoint["input"],
                example=datapoint["answer_icliniq"],
                response=datapoint[model],
                format_instructions=format_instruction
            ) 
            while True:
                try:
                    results = get_completion(
                        prompt=prompt,
                        temperature=args.temperature,
                    )
                    break
                except:
                    print(f"Error in getting completion for {model} with datapoint num {j}")
                    print(f"Retrying in {DELAY} second(s)...")
                    time.sleep(DELAY) # to avoid the API call limit
            scores = output_parser.parse(results)   
            scores_list[i].append(scores) 
        scores_averaged = get_average_scores(scores_list[i])
        scores_final[model] = scores_averaged
    
    jdump(scores_final, args.evaluation_results)
    print("=============================================")
    print("Evaluation results saved to {}".format(args.evaluation_results))


if __name__ == "__main__":
    main()
