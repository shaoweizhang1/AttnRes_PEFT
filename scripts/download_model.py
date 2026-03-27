import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.downloader import ModelLoader


def main():
    model_loader = ModelLoader()
    model_loader.run()


if __name__ == "__main__":
    main()
