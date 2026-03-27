import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.trainer import TrainerRunner, build_parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    trainer_runner = TrainerRunner(args)
    trainer_runner.run()


if __name__ == "__main__":
    main()
