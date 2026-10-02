"""Görüntü+soru ve yalnızca cevap tokenlarına uygulanan eğitim loss maskesi."""
import torch
from PIL import Image
from model import PROMPT

def encode(processor, image_path, answer=None, question=PROMPT):
    messages = [{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    inputs = processor(text=prompt, images=[image], return_tensors='pt')
    if answer is None:
        return inputs
    messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]})
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full = processor(text=full_text, images=[image], return_tensors='pt')
    prompt_length = inputs['input_ids'].shape[1]
    if not torch.equal(full['input_ids'][:, :prompt_length], inputs['input_ids']):
        raise ValueError('Sohbet şablonu öneki uyuşmuyor; cevap maskesi güvenli oluşturulamadı.')
    labels = full['input_ids'].clone()
    labels[:, :prompt_length] = -100  # Yalnızca doğru cevabı öğren.
    if (labels != -100).sum() == 0:
        raise ValueError('Öğrenilecek cevap tokenı bulunamadı.')
    full['labels'] = labels
    return full
