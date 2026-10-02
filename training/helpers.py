import json
import math
import random
from collections import defaultdict
import torch
from display import parse_answer
from .modeling import encode
from .metrics import label_sets, summarize

def balanced_epoch(rows, size, seed):
    """Her üçlü için en az 8 mevcut örneği koru, kalanı ağırlıklı seç; tekrar yok."""
    rng=random.Random(seed)
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        for key in label_sets(json.loads(row['answer']))['triplet']:
            groups[key].append(i)
    selected=set()
    for key,indices in sorted(groups.items()):
        selected.update(rng.sample(indices,min(8,len(indices))))
    size=min(size,len(rows))
    if size<len(selected):raise ValueError(f'epoch_samples en az {len(selected)} olmalı.')
    # Nadir sınıflara daha fazla şans ver, fakat ağırlığı 8 ile sınırla.
    counts={key:len(indices) for key,indices in groups.items()}
    largest=max(counts.values(), default=1)
    ranked=[]
    for i,row in enumerate(rows):
        if i in selected:continue
        labels=label_sets(json.loads(row['answer']))['triplet']
        weight=max([min(8.,math.sqrt(largest/counts[key])) for key in labels],default=1.)
        ranked.append((math.log(max(rng.random(),1e-12))/weight,i))
    ranked.sort(reverse=True)
    selected.update(i for _,i in ranked[:size-len(selected)])
    result=[rows[i] for i in sorted(selected)]
    rng.shuffle(result)
    return result

def validate_generation(model,processor,rows,target_device,categories,max_tokens=256):
    model.eval();results=[]
    vocabulary={task:set(categories[task].values()) for task in ('instrument','triplet','verb','target')}
    for index,row in enumerate(rows,1):
        inputs=encode(processor,row['image']).to(target_device)
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False)
        tokens=output[0,inputs['input_ids'].shape[1]:]
        raw=processor.decode(tokens,skip_special_tokens=True)
        try:prediction=parse_answer(raw)
        except ValueError:prediction=None
        results.append({'image':row['image'],'video':row['video'],'truth':json.loads(row['answer']),
                        'prediction':prediction,'raw':raw,'unknown_classes':{task:sorted(values-vocabulary[task]) for task,values in label_sets(prediction).items() if values-vocabulary[task]} if prediction is not None else {},'limit_hit':len(tokens)>=max_tokens})
        if index%20==0:print(f'Doğrulama: {index}/{len(rows)}',flush=True)
    summary,classes=summarize(results)
    # Destekli üçlülerde macro F1: sık sınıfların ortalamayı tek başına taşımasını azaltır.
    values=[2*c['tp']/(2*c['tp']+c['fp']+c['fn']) for c in classes['triplet'].values() if c['tp']+c['fn']>0]
    macro=sum(values)/len(values) if values else 0.
    micro=summary['micro_metrics']['triplet']['f1'] or 0.
    summary['supported_triplet_macro_f1']=macro
    summary['supported_triplet_classes']=len(values)
    summary['selection_score']=(micro+macro)/2
    return summary,results
