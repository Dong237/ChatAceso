import os
import sys
sys.path.append("/cluster/home/Aceso")

import warnings
from tqdm import tqdm
from tools.model import ChatLLM
from tools.utils import parse_args, jload, jdump
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

STEPS = ["step1", "step2", "step3"]
QUESTION_WITH_IMAGES="question_with_images.json"

TEMPLATE_STRING_LLM_EXAMINEE_NO_CONTEXT = "{question}, choose the correct answer from the following options: '''{options}'''. The Answer to the question is: "
TEMPLATE_STRING_LLM_EXAMINEE_WITH_CONTEXT = "Given the following context: {context} \n Answer the question:" + TEMPLATE_STRING_LLM_EXAMINEE_NO_CONTEXT
TEMPLATE_STRING_LLM_EXTRACTOR = """
The following texts are the answer of a single choice question given by a Large Language Model. Extract the answer as a single captital letter, if you are not able to find the answer, return '</404>'.

The Answer: '''{answer}'''.

{format_instructions}
""" 

RESPONSE_SCHEMA = ResponseSchema(
    name="answer", 
    type="string",
    description="The answer to the question in the form of a capital letter. For example: 'A'"
)
OUTPUT_PARSER = StructuredOutputParser.from_response_schemas([RESPONSE_SCHEMA])
FORMAT_INSTRCTION = OUTPUT_PARSER.get_format_instructions()


INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
# TODO: PYTHONPATH hacks are never a good idea. clean this up later
sys.path.append(os.path.join(INFERENCE_DIR, '..'))


## evaluation data processing for USMLE
def _remove_q_with_images(data_dir):
    idx_with_images = jload(os.path.join(
        data_dir, QUESTION_WITH_IMAGES
        )
        )
    steps_all = {}
    for step in STEPS:
        fstep = step+".json"
        questions = jload(os.path.join(data_dir, fstep))
        questions_no_images = [
            element for element in questions if element["no"] not in idx_with_images[step]
            ]
        steps_all.update({step: questions_no_images})
    return steps_all

def get_exam_for_step(dir, step: str):
    steps_all = _remove_q_with_images(dir)
    return steps_all[step]


## evaluation for USMLE
def accuracy(data_dir, step_all_answers, verbose=True) -> dict:
    accuracies = {}
    for step in STEPS:
        answers = step_all_answers[step]
        solutions = jload(
            os.path.join(data_dir, f"{step}_solutions.json")
            )
        correct_answers = {
            no: answers[no] 
            for no in answers.keys() 
            if no in solutions.keys() and answers[no] == solutions[no]
            }
        accuracy = len(correct_answers)/len(answers)
        accuracies.update({step: round(accuracy, 4)})
        if verbose:
            print(f"{step} Accuracy:" "{:.2f}%".format(accuracy * 100))
            print("--------------------")
    return accuracies

        
def usmle_evaluation(
        data_dir, 
        evaluation_results, 
        chain, 
        use_retireval, 
        wiki_data_dir, 
        use_extractor
        ):
    
    if use_retireval:
        print("Loading Wikipedia index...")
        wiki_index = WikipediaIndex(wiki_data_dir=wiki_data_dir)

    step_all_answers = dict()
    for step in STEPS:
        exam = get_exam_for_step(data_dir, step)
        answers = {}
        for query in tqdm(exam, desc=f"evaluating questions in {step}"):
            if use_retireval:
                input_ = {
                    "context": wiki_index.search(query["question"]), 
                    "question":query["question"], 
                    "options": query["options"]
                    }
            else:
                input_ = {
                    "question":query["question"], 
                    "options": query["options"]
                    }
            try:
                response =chain.run(input_)
            except:
                print(f"failed generating response for {step} question num. {query['no']}")
                response = """
                    ```json
                    {"answer": "GENERATION ERROR"}
                    ``` 
                """
            if use_extractor:
                try:
                    # may encounter failure of output format (very rare for GPT-3.5)
                    response = OUTPUT_PARSER.parse(response).get(
                        "answer", "EXTRACTION FAILED"
                        )
                except:
                    # failure of parser (still rare)
                    warnings.warn(f"Extractor failed at following instructions for {step}, question num. {query['no']}, please extract the answer manually")
            answers.update({str(query["no"]): response})
        step_all_answers.update({step: answers})

    accuracies = accuracy(data_dir, step_all_answers)
    print(accuracies)

    step_all_answers.update({"accuracies": accuracies})
    jdump(step_all_answers, os.path.join(data_dir, evaluation_results))

    print("==================================================================")
    print(f"Evaluation process finished with answers saved to {evaluation_results}")


def main():

    args = parse_args()
    
    inference_config = {
        'max_new_tokens': args.max_tokens,
        'do_sample': args.sample,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'repetition_penalty': args.repetition_penalty,
    }
    llm_examinee = ChatLLM(
        model_name_or_path=args.model, 
        cache_dir=args.cache_dir,
        peft=args.peft, 
        peft_weights=args.peft_weights, 
        gpu_id=args.gpu_id, 
        gpu_vram=args.gpu_vram, 
        cpu_ram=args.cpu_ram,
        inference_config=inference_config
        )
    
    if args.use_retrieval:
        prompt_examinee = PromptTemplate(
            template=TEMPLATE_STRING_LLM_EXAMINEE_WITH_CONTEXT,
            input_variables=["context", "question", "options"]
        )
    else:
        prompt_examinee = PromptTemplate(
            template=TEMPLATE_STRING_LLM_EXAMINEE_NO_CONTEXT,
            input_variables=["question", "options"]
        )
    examinee_chain = LLMChain(
        llm=llm_examinee,
        prompt=prompt_examinee,
        output_key="answer",
        verbose=False,
        )
    
    if args.use_extractor:
        print("======================")
        print("Using extractor model")
        print("======================")
        llm_extractor = OpenAI(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model=args.extractor_model,
            client=None,
            )
        extractor_chain = LLMChain(
            llm=llm_extractor,
            prompt=PromptTemplate(
                template=TEMPLATE_STRING_LLM_EXTRACTOR,
                input_variables=["answer"],
                partial_variables={"format_instructions": FORMAT_INSTRCTION}
                ),
            output_key="dict",
            verbose=False,
        )
        examinee_extractor_chain = SequentialChain(
            chains=[examinee_chain, extractor_chain],
            input_variables=["context", "question", "options"] \
                if args.use_retrieval else ["question", "options"],
            verbose=False,
            )
    
    usmle_evaluation(
        data_dir=args.eval_data_path, 
        evaluation_results=args.evaluation_results, 
        chain=examinee_extractor_chain if args.use_extractor else examinee_chain, 
        use_retireval=args.use_retrieval,
        wiki_data_dir=args.retrieval_data_path,
        use_extractor=args.use_extractor
    )


if __name__ == '__main__':
    main()