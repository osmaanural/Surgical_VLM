"""Dengeli veri + görsel bağlantı eğitimi + üretilmiş cevaplarla model seçimi."""
import json
import math
import random
import time
import torch
from peft import LoraConfig,get_peft_model
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from model import choose_device
from .modeling import encode
from .data import read_records
from .helpers import balanced_epoch,validate_generation
from .metrics import sample_records


def build_model(cfg):
    processor=AutoProcessor.from_pretrained(cfg['base_model'],size={'longest_edge':512})
    model=AutoModelForImageTextToText.from_pretrained(cfg['base_model'],dtype=torch.float32,attn_implementation='eager')
    targets=[name for name,layer in model.named_modules() if isinstance(layer,torch.nn.Linear)
             and 'text_model' in name and name.endswith(('q_proj','v_proj'))]
    if not targets or not any(name.endswith('connector') for name,_ in model.named_modules()):
        raise ValueError('Beklenen VLM katmanları bulunamadı.')
    model=get_peft_model(model,LoraConfig(r=cfg['lora_rank'],lora_alpha=16,lora_dropout=0.05,
        target_modules=targets,modules_to_save=['connector'],bias='none'))
    return processor,model


def train(args):
    cfg=json.loads(args.config.read_text(encoding="utf-8"))
    for key in ('epochs','epoch_samples','gradient_accumulation','validation_per_video','patience'):
        if cfg[key]<1:raise ValueError(f'{key} en az 1 olmalı.')
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    target_device=choose_device(torch,args.device)
    if target_device=='cpu' and not args.check:raise RuntimeError('GPU yok; eğitim başlatılmadı. Veri/ileri geçiş kontrolü için --check kullanabilirsiniz.')
    train_rows,categories=read_records(args.data_dir,'train')
    val_all,val_categories=read_records(args.data_dir,'val')
    if categories != val_categories:raise ValueError('Eğitim/doğrulama sınıf eşlemeleri farklı.')
    val_rows=sample_records(val_all,cfg['validation_per_video'],cfg['seed'])
    first_epoch=balanced_epoch(train_rows,cfg['epoch_samples'],cfg['seed'])
    processor,model=build_model(cfg);model.to(target_device)
    if args.check:
        sample=first_epoch[0]
        inputs=encode(processor,sample['image'],sample['answer']).to(target_device)
        model.eval()
        with torch.inference_mode():loss=model(**inputs).loss.item()
        if not math.isfinite(loss):raise ValueError('İleri geçiş loss değeri sonlu değil.')
        print(f'Kontrol başarılı: {len(train_rows)} eğitim karesi, {len(val_all)} doğrulama karesi, loss={loss:.4f}')
        print('Eğitim veya ağırlık güncellemesi yapılmadı.')
        return
    model.print_trainable_parameters()
    connector=[];lora=[]
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            (connector if 'connector' in name else lora).append(parameter)
    optimizer=torch.optim.AdamW([{'params':lora,'lr':cfg['learning_rate']},
                                {'params':connector,'lr':cfg['connector_learning_rate']}])
    run=args.output/time.strftime('%Y%m%d-%H%M%S');run.mkdir(parents=True,exist_ok=False)
    (run/'config.json').write_text(json.dumps({'training':cfg,'data_dir':str(args.data_dir.resolve())},indent=2))
    (run/'validation_images.json').write_text(json.dumps([r['image'] for r in val_rows]))
    print(f'Yeni eğitim: {run}',flush=True)
    best=-1.;stale=0;accumulation=cfg['gradient_accumulation']
    for epoch in range(cfg['epochs']):
        rows=balanced_epoch(train_rows,cfg['epoch_samples'],cfg['seed']+epoch)
        (run/f'epoch_{epoch+1}_images.json').write_text(json.dumps([r['image'] for r in rows]))
        model.train();optimizer.zero_grad();total=0.;start=time.time()
        for index,row in enumerate(rows):
            inputs=encode(processor,row['image'],row['answer']).to(target_device)
            loss=model(**inputs).loss
            if not torch.isfinite(loss):raise RuntimeError('Loss sonlu değil; eğitim durdu.')
            group_start=index//accumulation*accumulation
            (loss/min(accumulation,len(rows)-group_start)).backward()
            total+=loss.item()
            if (index+1)%accumulation==0 or index+1==len(rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();optimizer.zero_grad()
            if (index+1)%50==0:print(f'Epoch {epoch+1}: {index+1}/{len(rows)}, loss={loss.item():.4f}',flush=True)
        model.save_pretrained(run/'last');processor.save_pretrained(run/'last')
        metrics,predictions=validate_generation(model,processor,val_rows,target_device,categories)
        (run/f'val_epoch_{epoch+1}.json').write_text(json.dumps({'metrics':metrics,'predictions':predictions},indent=2))
        score=metrics['selection_score']
        if score>best:
            best=score;stale=0
            model.save_pretrained(run/'best');processor.save_pretrained(run/'best')
        else:stale+=1
        record={'epoch':epoch+1,'train_loss':total/len(rows),'validation':metrics,'seconds':time.time()-start}
        with (run/'history.jsonl').open('a') as file:file.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if stale>=cfg['patience']:
            print('Doğrulama skoru gelişmedi; erken durduruldu.');break
    print('Tamamlandı. Test verisi kullanılmadı. Yeni adaptör:',run/'best')



