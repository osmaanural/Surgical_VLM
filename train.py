"""İsteğe bağlı yeni LoRA + connector eğitimi; hazır model/ klasörüne dokunmaz."""
import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path, help='CholecT50 ana dizini (labels/ ve videos/ içeren)')
    parser.add_argument('--config', type=Path, default=BASE/'training/config.json')
    parser.add_argument('--output', type=Path, default=BASE/'training_runs')
    parser.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    parser.add_argument('--check', action='store_true', help='Veri ve ileri geçiş kontrolü; eğitim yapmaz')
    args = parser.parse_args()
    from training.engine import train
    train(args)


if __name__ == '__main__':
    main()
