import os
import torch
from transformers import AutoTokenizer, GenerationConfig
from auto_gptq.modeling import MistralGPTQForCausalLM

import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from dotenv import load_dotenv

from conversation import Conversation

load_dotenv()


MODEL_NAME = './models/mistral-saiga-yakutia-GPTQ'

model = MistralGPTQForCausalLM.from_quantized(
    MODEL_NAME,
    device="cuda:0",
    use_triton=False,
    inject_fused_mlp=True,
    inject_fused_attention=True,
    )

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("Model loaded successfully!")

def get_dialog_id(event):
    if event.from_user:
        return event.user_id
    if event.from_chat:
        return event.chat_id


# VK
VK_API_KEY = str(os.getenv("VK_API_KEY"))
vk_session = vk_api.VkApi(token=VK_API_KEY)
vk_api = vk_session.get_api()
longpoll = VkLongPoll(vk_session)
dialogs = {}
for event in longpoll.listen():
    if event.type == VkEventType.MESSAGE_NEW and event.to_me and event.text:
        dialog_id = get_dialog_id(event)
        if dialog_id:
            if dialogs.get(dialog_id):
                dialog = dialogs[dialog_id]
            else:
                dialog = Conversation()
                dialogs[dialog_id] = dialog
            answer = dialog.generate(model, tokenizer, generation_config, event.text)
            if event.from_user:  # Если написали в ЛС
                vk_api.messages.send(
                    user_id=event.user_id,
                    message=answer,
                    random_id=0,
                )
            elif event.from_chat:  # Если написали в Беседе
                vk_api.messages.send(
                    chat_id=event.chat_id,
                    message=answer,
                    random_id=0,
                )


if __name__ == '__main__':
    print("OK")
