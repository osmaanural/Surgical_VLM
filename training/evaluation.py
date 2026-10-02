"""test/ görüntülerini dene; --data-dir verilirse gerçek etiketlerle skor hesapla."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=run.BASE/'test', help='Etiketsiz görüntü klasörü')
    parser.add_argument('--data-dir', type=Path, help='Skor hesaplamak için CholecT50 dizini')
    parser.add_argument('--split', choices=['val','test'], default='test')
    parser.add_argument('--per-video', type=int, default=20, help='Skorlu modda video başına kare; 0=tümü')
    parser.add_argument('--adapter', type=Path, default=run.BASE/'model')
    parser.add_argument('--output', type=Path, default=run.BASE/'results')
    parser.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()
    if args.per_video < 0 or args.max_new_tokens < 1:
        parser.error('Kare/token sayısı geçersiz.')
    if args.data_dir is None:
        if not args.input.is_dir():parser.error('--input bir klasör olmalı; tek görüntü için predict.py kullanın.')
        return run.main(['--input',str(args.input),'--adapter',str(args.adapter),'--output',str(args.output),
                         '--device',args.device,'--max-new-tokens',str(args.max_new_tokens)])
    from model import SurgicalVLM
    from training.data import read_records
    from training.metrics import sample_records
    from training.helpers import validate_generation
    rows,categories = read_records(args.data_dir,args.split)
    rows = sample_records(rows,args.per_video,42)
    vlm = SurgicalVLM(args.adapter,args.device)
    metrics,predictions = validate_generation(vlm.model,vlm.processor,rows,vlm.device,categories,args.max_new_tokens)
    folder = args.output/('evaluation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True,exist_ok=False)
    report={'split':args.split,'per_video':args.per_video,'adapter':str(args.adapter.resolve()),
            'metrics':metrics,'predictions':predictions}
    (folder/'evaluation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(metrics,indent=2,ensure_ascii=False))
    print('Sonuç:',folder/'evaluation.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
