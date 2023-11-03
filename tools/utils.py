import os
import io
import json
import random
import argparse
from tqdm import tqdm


MEANINGLESS_WORDS = ['<pad>', '</s>', '<s>']


# arguments
def parse_args():
    parser = argparse.ArgumentParser()
    # model auguments
    parser.add_argument(
        '--model',
        default=f"meta-llama/Llama-2-7b-chat-hf",
        help='name/path of the model'
    )
    parser.add_argument(
        '--cache-dir',
        default="/scratch/huggingface",
        help='name/path of the cache directory'
    )
    parser.add_argument(
        '--peft',
        default=False,
        help='whether the model was trained with PEFT, if yes, please provide the path to the PEFT weights'
    )
    parser.add_argument(
        '--peft-weights',
        default=f"/peft_weights",
        help='name/path of peft-weights'
    )
    # hardware arguments
    parser.add_argument(
        '--gpu-id',
        default=0,
        type=int,
        help='the ID of the GPU to run on'
    )
    parser.add_argument(
        '-g',
        '--gpu-vram',
        action='store',
        help='max VRAM to allocate per GPU',
        nargs='+',
        required=False,
    )
    parser.add_argument(
        '-r',
        '--cpu-ram',
        default=None,
        type=int,
        help='max CPU RAM to allocate',
        required=False
    )
    # inference arguments
    parser.add_argument(
        '--max-tokens',
        default=128,
        type=int,
        help='the maximum number of tokens to generate'
    )
    parser.add_argument(
        '--sample',
        default=True,
        help='indicates whether to sample'
    )
    parser.add_argument(
        '--temperature',
        default=0.6,
        type=float,
        help='temperature for the LM'
    )
    parser.add_argument(
        '--top-k',
        default=40,
        type=int,
        help='top-k for the LM'
    )
    parser.add_argument(
        '--repetition-penalty',
        default=1.0,
        type=float,
        help='repetition penalty for the generation'
        )
    # evaluation arguments
    parser.add_argument(
        '--eval-data-path',
        type=str,
        help='path to the evaluation data'
    )
    parser.add_argument(
        '--use-retrieval',
        action="store_true",
        help='whether to use retrieval from knowledge base as context'
    )
    parser.add_argument(
        '--retrieval-data-path',
        default="/scratch/files",
        help='path to the retrieval data directory'
    )
    parser.add_argument(
        '--summarizer-device',
        default="cuda:0",
        help='device to run the summarizer on'
    )
    parser.add_argument(
        '--evaluation-results',
        default="evaluation_results.json",
        help='file name for storing evaluation results'
    )
    # for usmle eval only
    parser.add_argument(
        '--use-extractor',
        action='store_true',
        help='whether to use an extractor model to extract answers'
    )
    parser.add_argument(
        '--extractor-model',
        default="text-davinci-003",
        help='the instruction-following model (from OpenAI) used to help extracting answers'
    )
    # for metrics and gpt4 eval
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help='random seed for sampling'
    )
    parser.add_argument(
        '--sample-size',
        default=100,
        type=int,
        help='number of samples to draw'
    )
    parser.add_argument(
        '--new-evaluation',
        action='store_true',
        help='whether to force resampling the data and performing a new evaluation'
    )
    parser.add_argument(
        '--inference-results',
        default="inference_results.json",
        help='file name for storing inference results (will be read for evaluation if it exists)'
    )
    # for gpt4 eval only
    parser.add_argument(
        '--model-version1',
        default=f"meta-llama/Llama-2-7b-chat-hf",
        help='name/path of the model'
    )
    parser.add_argument(
        '--gpu-id-base',
        default=0,
        type=int,
        help='the ID of the GPU to run on'
    )
    parser.add_argument(
        '-g-base',
        '--gpu-vram-base',
        action='store',
        help='max VRAM to allocate per GPU',
        nargs='+',
        required=False,
    )
    parser.add_argument(
        '--gpu-id-version1',
        default=0,
        type=int,
        help='the ID of the GPU to run on'
    )
    parser.add_argument(
        '-g-version1',
        '--gpu-vram-version1',
        action='store',
        help='max VRAM to allocate per GPU',
        nargs='+',
        required=False,
    )
    parser.add_argument(
        '--gpu-id-version2',
        default=0,
        type=int,
        help='the ID of the GPU to run on'
    )
    parser.add_argument(
        '-g-version2',
        '--gpu-vram-version2',
        action='store',
        help='max VRAM to allocate per GPU',
        nargs='+',
        required=False,
    )

    args = parser.parse_args()
    return args


## general loading and saving
def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def clean_response(response):
    for word in MEANINGLESS_WORDS:
        response = response.replace(word, "")
    response = response.strip("\n")
    return response


def get_max_memory(gpu_vram, cpu_ram) -> dict:
    if gpu_vram is None:
        max_memory = None
    else:
        max_memory = {}
        for i in range(len(gpu_vram)):
            # assign CUDA ID as label and XGiB as value
            max_memory[int(gpu_vram[i].split(':')[0])] = f"{gpu_vram[i].split(':')[1]}GiB"

        if cpu_ram is not None:
            # add cpu to max-memory if given
            max_memory['cpu'] = f"{int(cpu_ram)}GiB"
    return max_memory


# function for performing infernce on data batch 
def collect_answers(
    model, 
    args, 
    datapoints,
    answer_key="answer_aceso"):
    inference_config = {
            'max_new_tokens': args.max_tokens,
            'do_sample': args.sample,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'repetition_penalty': args.repetition_penalty,
        }   
    for datapoint in tqdm(datapoints, desc="Collecting model answers:"):
        datapoint[answer_key] = model.do_inference(
            prompt=datapoint["input"], 
            **inference_config,
            )
    print("Finished collection.")
    return datapoints