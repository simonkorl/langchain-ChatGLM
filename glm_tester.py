from transformers import AutoModel, AutoTokenizer
from typing import List
import json

class GLMTester:
    tokenizer = None
    model = None
    test_history = []
    max_length = 10000
    def __init__(self, model_name="THUDM/chatglm-6b"):
      self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                            trust_remote_code=True)
      self.model = AutoModel.from_pretrained(model_name,
                trust_remote_code = True).half().cuda()
      self.test_history = []

    def chat(self, query: str, history: List[List[str]] = []):
        '''
        # chatglm-6b
        chat(tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1, do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs)
        # chatglm2-6b
        chat(tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None, **kwargs)
        '''
        return self.model.chat(self.tokenizer, query, history, max_length=self.max_length)
    def test(self, template: str, question: str, context: str = "",
             history: List[List[str]] = [],
             store=True, print_response=True):
        query = template.replace("{context}", context).replace("{question}", question)
        response, history_after = self.chat(query, history)
        if print_response:
            print(response)
        if store:
            self.test_history.append({
            "question": question,
            "template": template,
            "context": context,
            "answer": response,
            "history": history,
            "history_after": history_after
            })
        return response, history_after

    def save_test_history(self, filename: str = "test_history.json"):
        with open(filename, 'w') as f:
            json.dump(self.test_history, f)

# glm = GLMTester("THUDM/chatglm-6b")
# glm2 = GLMTester("THUDM/chatglm2-6b")