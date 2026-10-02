"""CholecT50'yi yerinde okur; görüntüleri kopyalamaz."""
import json
from pathlib import Path


def read_records(root, part):
    root = Path(root).expanduser().resolve()
    splits = json.loads(Path(__file__).with_name('splits.json').read_text())
    all_ids = sum(splits.values(), [])
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Eğitim/doğrulama/test videoları çakışıyor.')
    rows, categories = [], None
    for video in splits[part]:
        path = root / 'labels' / f'{video}.json'
        if not path.is_file():
            raise FileNotFoundError(f'Etiket bulunamadı: {path}. --data-dir CholecT50 ana klasörünü göstermeli.')
        doc = json.loads(path.read_text(encoding='utf-8'))
        if categories is None:
            categories = doc['categories']
        if categories != doc['categories']:
            raise ValueError(f'Sınıf eşlemesi farklı: {video}')
        for frame in sorted(doc['annotations'], key=int):
            instruments, triples = set(), set()
            for annotation in doc['annotations'][frame]:
                if len(annotation) != 15:
                    raise ValueError(f'Geçersiz etiket: {video}/{frame}')
                if annotation[1] >= 0:
                    instruments.add(categories['instrument'][str(annotation[1])])
                if annotation[0] >= 0:
                    triples.add(categories['triplet'][str(annotation[0])])
            answer = {'instruments': sorted(instruments), 'triplets': [
                dict(zip(('instrument','verb','target'), triple.split(','))) for triple in sorted(triples)]}
            image = root / 'videos' / video / f'{int(frame):06d}.png'
            if not image.is_file():
                raise FileNotFoundError(image)
            rows.append({'image': str(image), 'video': video, 'answer': json.dumps(answer)})
    if not rows:
        raise ValueError('Veri kümesi boş.')
    return rows, categories
