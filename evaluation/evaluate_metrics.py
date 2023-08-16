import os
import sys
import random
sys.path.append("/cluster/home/Aceso")

import nltk
nltk.download('punkt')
import evaluate
from tqdm import tqdm
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.translate import gleu_score
from tools.utils import collect_answers, get_max_memory, parse_args, jdump, jload
from tools.model import ChatModel

ANSWERS_KEYS = ["answer_icliniq", "answer_chatgpt", "answer_chatdoctor", "answer_aceso"]

# function for calculating gleu
def compute_avg_gleu(pred_list, ref_list):
    total_score = 0
    for pred, ref in zip(pred_list, ref_list):
        pred_tokens = word_tokenize(pred)
        ref_tokens = word_tokenize(ref)
        total_score += gleu_score.sentence_gleu([ref_tokens], pred_tokens)
    return total_score / len(ref_list)


# function for calculating distinct-n
def distinct_n_corpus_level(corpus, n):
    def distinct_n_sentence_level(sentence, n):
        if len(sentence) == 0:
            return 0.0  # Prevent a zero division
        distinct_ngrams = Counter(ngrams(sentence, n))
        return len(distinct_ngrams) / len(sentence)
    return sum(distinct_n_sentence_level(sentence, n) for sentence in corpus) / len(corpus)


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
    
    answers_list = [[], [], [],[]]
    for i, answer_key in enumerate(ANSWERS_KEYS):
        for datapoint in datapoints:
            answers_list[i].append(datapoint[answer_key])

    final_scores = dict()
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    for i, answer_key in tqdm(enumerate(ANSWERS_KEYS), desc="Calculating scores for different models:"):
        if answer_key != "answer_icliniq":
            bleu_score = bleu.compute(
                predictions=answers_list[i], 
                references=answers_list[0], 
                max_order=4
                )
            rouge_score = rouge.compute(
                predictions=answers_list[i], 
                references=answers_list[0]
                )
            gleu_score = compute_avg_gleu(answers_list[i], answers_list[0])
            distinct_1 = distinct_n_corpus_level(answers_list[i], 1)
            distinct_2 = distinct_n_corpus_level(answers_list[i], 2)

            scores = {
                "bleu1": bleu_score["precisions"][0]*bleu_score["brevity_penalty"],
                "bleu2": bleu_score["precisions"][1]*bleu_score["brevity_penalty"],
                "bleu3": bleu_score["precisions"][2]*bleu_score["brevity_penalty"],
                "bleu4": bleu_score["precisions"][3]*bleu_score["brevity_penalty"],
                "gleu": gleu_score,
                "rouge1": rouge_score["rouge1"],
                "rouge2": rouge_score["rouge2"],
                "rougeL": rouge_score["rougeL"],
                "distinct1": distinct_1,
                "distinct2": distinct_2,
            }
            final_scores[answer_key] = scores

    jdump(final_scores, args.evaluation_results)
    print("Finished evaluation, results saved to: ", args.evaluation_results)


if __name__ == '__main__':
    main()