import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.downloader import DataLoader


def main():
    data_loader = DataLoader()
    data_loader.run()


if __name__ == "__main__":
    main()
