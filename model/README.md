# Eğitilmiş model

Bu klasörün tamamı kullanım dağıtımının bir parçasıdır; Git'e dahil edin.

- adapter_model.safetensors: LoRA ve eğitilmiş connector ağırlıkları.
- adapter_config.json: Temel model adı ve adaptör mimarisi.
- processor_config.json, tokenizer dosyaları, chat_template.jinja: Eğitimdeki ön işleme.

Temel VLM ağırlıkları bu klasörde değildir; uygulama ilk kullanımda indirir.
Model kaynağı ve ölçümleri: ../docs/MODEL.md
