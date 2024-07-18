import argparse

import numpy as np
import torch


def gen_data(batch_size, output):
    a = torch.randint(0, 50265, (batch_size, 384))
    a = a.numpy().astype(np.int64)
    a.tofile(output+"input_ids.bin")

    a = np.ones((batch_size, 384), dtype=np.int64)
    a.tofile(output+"input_mask.bin")

    a = np.zeros((batch_size, 384), dtype=np.int64)
    a.tofile(output+"token_type_ids.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data for RoBERTa model.")
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size for data generation"
    )
    parser.add_argument("--output_path", default="")

    args = parser.parse_args()

    gen_data(args.batch_size, args.output_path)