import re
import time
from tools.utils import clean_response

PRE_PROMPT = """\
Current Date: {}
Current Time: {}

"""

INSTRUCTION = """\
If you are a doctor named Aceso, interact with the person and answer the query. Always answer as helpfully as possible, while being safe.  

Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

SYSTEM_PROMPT = f'<s>[INST] <<SYS>>\n{INSTRUCTION}\n<</SYS>>\n\n'

class Conversation:
    def __init__(self):
        self.cur_date = time.strftime('%Y-%m-%d')
        self.cur_time = time.strftime('%H:%M:%S %p %Z')

        self._prompt = PRE_PROMPT.format(self.cur_date, self.cur_time)
        self._prompt += SYSTEM_PROMPT
        self._context = ""

    def push_context_turn(self, context):
        self._prompt += "If you don't know the answer of the question, don't make up an answer, use the information from the following context \n"
        self._prompt += f"<context>:\n {context}\n"
        self._context = context

    def push_human_turn(self, query):
        self._prompt += f"{query} {'[/INST]'} "

    def push_model_response(self, response):
        self._prompt += f"{response} {'</s><s>[INST]'} "