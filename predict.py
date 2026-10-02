"""Tek bir görüntü için Türkçe görüntülü tahmin."""
import argparse
from pathlib import Path
import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, help='Verilmezse test/ içindeki ilk görüntü kullanılır.')
    parser.add_argument('--adapter', type=Path, default=run.BASE/'model')
    parser.add_argument('--output', type=Path, default=run.BASE/'results')
    parser.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()
    if args.image is None:
        images = run.find_images(run.BASE / 'test')
        if not images:
            parser.error('test/ klasörüne bir PNG veya JPG görüntüsü koyun.')
        example = run.BASE / 'test' / 'ornek.png'
        args.image = example if example in images else images[0]
    print(f'Seçilen görüntü: {args.image}')
    if not args.image.is_file():
        parser.error('--image tek bir mevcut görüntü dosyası olmalı.')
    return run.main(['--input',str(args.image),'--adapter',str(args.adapter),'--output',str(args.output),
                     '--device',args.device,'--max-new-tokens',str(args.max_new_tokens)])


if __name__ == '__main__':
    raise SystemExit(main())
