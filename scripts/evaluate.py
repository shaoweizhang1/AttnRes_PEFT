import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluator import Evaluator


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./model")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", default="./results")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
