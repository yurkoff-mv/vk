from typing import Union

from auto_gptq.modeling import MistralGPTQForCausalLM
from transformers import MistralForCausalLM, LlamaTokenizerFast, GenerationConfig


class Conversation:
    system_prompt = ("Ты умный цифровой помощник Главы республики Саха (Якутия)."
                     "Ты разговариваешь с людьми и помогаешь им.")
    message_template = "<s>{role}\n{content}</s>\n"
    response_template = "<s>bot\n"
    max_length = 4096

    def __init__(self):
        self.messages = [{
            "role": "system",
            "content": self.system_prompt
        }]

    def add_user_message(self, message: str):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message: str):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += self.response_template
        return final_text.strip()

    def generate(self,
                 model: Union[MistralForCausalLM, MistralGPTQForCausalLM],
                 tokenizer: LlamaTokenizerFast,
                 generation_config: GenerationConfig,
                 question: str):
        self.add_user_message(question)
        tokens = tokenizer(self.get_prompt(), return_tensors="pt", add_special_tokens=False)
        # Remove last user and bot message
        if len(tokens.input_ids) > self.max_length:
            self.messages.pop(1)
            self.messages.pop(1)
            tokens = tokenizer(self.get_prompt(), return_tensors="pt", add_special_tokens=False)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        output_ids = model.generate(
            **tokens,
            generation_config=generation_config
        )[0]
        output_ids = output_ids[len(tokens["input_ids"][0]):]
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        answer = output.strip()
        self.add_bot_message(answer)
        return answer

