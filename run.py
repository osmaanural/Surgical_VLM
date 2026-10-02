"""test/ içindeki görüntüleri analiz eder, Türkçe sonuçları results/ içine kaydeder."""
import argparse
import json
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def find_images(source):
    if source.is_file():
        if source.suffix.lower() not in EXTENSIONS:
            raise ValueError('Desteklenen türler: PNG, JPG, JPEG, WEBP, BMP.')
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f'Girdi yolu bulunamadı: {source}')
    return sorted(path for path in source.rglob('*')
                  if path.is_file() and path.suffix.lower() in EXTENSIONS
                  and not any(part.startswith('.') for part in path.relative_to(source).parts))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=BASE / 'test', help='Görüntü klasörü veya tek görüntü')
    parser.add_argument('--output', type=Path, default=BASE / 'results', help='Sonuçların kaydedileceği klasör')
    parser.add_argument('--adapter', type=Path, default=BASE / 'model', help='Eğitilmiş adaptör klasörü')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens en az 1 olmalı.')
    source, output_root = args.input.resolve(), args.output.resolve()
    try:
        images = find_images(source)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if not images:
        print(f'Görüntü bulunamadı. PNG veya JPG dosyalarınızı buraya koyun: {source}')
        print('Ardından yeniden: python run.py')
        return 0
    if source.is_dir() and (output_root == source or source in output_root.parents):
        parser.error('Sonuç klasörü test/girdi klasörünün içinde olamaz.')

    from model import SurgicalVLM
    from display import parse_answer, save_visual
    print(f'{len(images)} görüntü bulundu. Model yükleniyor...', flush=True)
    try:
        vlm = SurgicalVLM(args.adapter, args.device)
    except (ValueError, FileNotFoundError, OSError) as error:
        print(f'Model yüklenemedi: {error}')
        return 1
    print(f'Cihaz: {vlm.device}', flush=True)
    run_folder = output_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_folder.mkdir(parents=True, exist_ok=False)
    summary = []
    for index, image in enumerate(images, 1):
        relative = image.relative_to(source) if source.is_dir() else Path(image.name)
        # a.jpg ile a.png aynı ada yazılmaz: a.jpg.png ve a.png.png oluşturulur.
        destination = run_folder / relative.parent / relative.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = {'image': str(relative), 'status': 'ok'}
        try:
            raw, limit_hit = vlm.predict(image, args.max_new_tokens)
            Path(str(destination) + '.raw.txt').write_text(raw, encoding='utf-8')
            result = parse_answer(raw)
            visual = Path(str(destination) + '.png')
            translated = save_visual(image, result, visual, True)
            details = {'prediction': result, 'turkish': translated, 'generation_limit_hit': limit_hit,
                       'note': 'Model tahmini; doğrulanmış etiket değildir.'}
            Path(str(destination) + '.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
            record['visual'] = str(visual.relative_to(run_folder))
            record['generation_limit_hit'] = limit_hit
        except (ValueError, OSError) as error:
            # Bozuk görüntü/yanıt diğer görüntülerin işlenmesini engellemez.
            record.update(status='error', error=str(error))
        summary.append(record)
        (run_folder / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[{index}/{len(images)}] {relative}: {record["status"]}', flush=True)
    errors = sum(row['status'] == 'error' for row in summary)
    print(f'Tamamlandı. Başarılı: {len(images)-errors}, hatalı: {errors}')
    print(f'Sonuç klasörü: {run_folder}')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
