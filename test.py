"""test/ klasöründeki bütün görüntüleri çalıştırır."""
import sys
import run


def main():
    # Etiketli skor hesabı isteyenler için eski seçenek korunur.
    labeled = any(arg == '--data-dir' or arg.startswith('--data-dir=') for arg in sys.argv[1:])
    if labeled:
        from training.evaluation import main as evaluate
        return evaluate()
    return run.main()


if __name__ == '__main__':
    raise SystemExit(main())
