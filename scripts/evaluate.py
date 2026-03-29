import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluator import Evaluator, build_parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
