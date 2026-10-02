"""Hazır VLM'yi ve bu projeyle gelen eğitilmiş adaptörü yükler."""
import json
from pathlib import Path

PROMPT = (
    'Identify the surgical instruments and instrument-verb-target triplets in this image. '
    'Return only JSON with keys instruments (list of names) and triplets '
    '(list of objects with instrument, verb, target). Use CholecT50 English class names. '
    'Do not infer surgical intent or physical measurements.'
)


def choose_device(torch, requested):
    available = {'cpu': True, 'cuda': torch.cuda.is_available(),
                 'mps': torch.backends.mps.is_available()}
    if requested == 'auto':
        return next(name for name in ('cuda', 'mps', 'cpu') if available[name])
    if not available[requested]:
        raise ValueError(f'{requested} cihazı bulunamadı. --device cpu kullanabilirsiniz.')
    return requested


class SurgicalVLM:
    def __init__(self, adapter_dir, device='auto'):
        # Boş test klasöründe büyük paketleri veya modeli yüklemeyiz.
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import PeftModel

        adapter_dir = Path(adapter_dir).resolve()
        for name in ('adapter_config.json', 'adapter_model.safetensors', 'processor_config.json'):
            if not (adapter_dir / name).is_file():
                raise FileNotFoundError(f'Model dosyası eksik: {adapter_dir / name}. model/ klasörünü eksiksiz indirin.')
        weights = adapter_dir / 'adapter_model.safetensors'
        with weights.open('rb') as file:
            if file.read(80).startswith(b'version https://git-lfs.github.com'):
                raise ValueError('Gerçek ağırlık yerine Git LFS işaretçisi var. git lfs pull çalıştırın.')
        adapter_config = json.loads((adapter_dir / 'adapter_config.json').read_text(encoding='utf-8'))
        self.base_model = adapter_config['base_model_name_or_path']
        self.device = choose_device(torch, device)
        self.torch = torch
        # Eğitimde kaydedilen görüntü işlemcisi ve tokenizer aynı kalır.
        self.processor = AutoProcessor.from_pretrained(str(adapter_dir), size={'longest_edge': 512})
        base = AutoModelForImageTextToText.from_pretrained(
            self.base_model, dtype=torch.float32, attn_implementation='eager')
        self.model = PeftModel.from_pretrained(base, str(adapter_dir)).to(self.device).eval()

    def predict(self, image_path, max_tokens=256):
        from PIL import Image
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        inputs = self.processor(text=prompt, images=[image], return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        generated = output[0, inputs['input_ids'].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True), len(generated) >= max_tokens
