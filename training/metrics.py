from collections import Counter, defaultdict
import random

def label_sets(answer):
    triples = {tuple(row[key] for key in ('instrument', 'verb', 'target')) for row in answer['triplets']}
    return {'instrument': set(answer['instruments']),
            'triplet': {','.join(row) for row in triples},
            'target': {row[2] for row in triples}, 'verb': {row[1] for row in triples}}

def scores(tp, fp, fn):
    # Payda yoksa ölçüm tanımsızdır; JSON'da null gösterilir.
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}

def summarize(rows):
    totals = {task: Counter() for task in ('instrument', 'triplet', 'target', 'verb')}
    classes = {task: defaultdict(Counter) for task in totals}
    exact = Counter()
    for row in rows:
        truth = label_sets(row['truth'])
        predicted = label_sets(row['prediction'] or {'instruments': [], 'triplets': []})
        for task in totals:
            t, p = truth[task], predicted[task]
            totals[task].update(tp=len(t & p), fp=len(p-t), fn=len(t-p))
            if row['prediction'] is not None and t == p:
                exact[task] += 1
            for label in t | p:
                classes[task][label].update(tp=int(label in t and label in p),
                    fp=int(label in p and label not in t), fn=int(label in t and label not in p))
    metrics = {}
    for task, count in totals.items():
        metrics[task] = scores(count['tp'],count['fp'],count['fn'])
        metrics[task]['exact_match'] = exact[task]/len(rows) if rows else None
    return {'examples':len(rows), 'invalid_json':sum(r['prediction'] is None for r in rows),
            'unknown_class_responses':sum(bool(r['unknown_classes']) for r in rows),
            'generation_limit_hits':sum(r['limit_hit'] for r in rows),
            'micro_metrics':metrics}, classes

def sample_records(records, per_video, seed):
    groups = defaultdict(list)
    for row in records: groups[row['video']].append(row)
    selected = []
    rng = random.Random(seed)
    for video, group in sorted(groups.items()):
        selected.extend(rng.sample(group, min(per_video,len(group))) if per_video else group)
    return sorted(selected, key=lambda row:(row['video'],row['image']))
