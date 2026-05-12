#!/usr/bin/env python3
"""Extract a fractional subset of an HDF5 file, preserving its exact structure."""

import argparse
import os
import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="Extract a fractional subset of an HDF5 file, preserving its structure.")
    parser.add_argument("-f", "--file", required=True, help="Input HDF5 file")
    parser.add_argument("-r", "--ratio", type=float, required=True, help="Fraction of rows to keep (e.g. 0.02 for 2%%)")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.file
    ratio = args.ratio

    if not 0 < ratio <= 1:
        raise ValueError("ratio must be between 0 (exclusive) and 1 (inclusive)")

    # Derive output path by inserting ".small" before the extension
    base, ext = os.path.splitext(input_path)
    output_path = base + ".small" + ext

    with h5py.File(input_path, "r") as input_file, h5py.File(output_path, "w") as output_file:
        for name, dataset in input_file.items():
            if dataset.shape == (1,) or dataset.shape == ():
                # Scalar/config datasets are small — copy them in full using [()] to read all data
                output_file.create_dataset(name, data=dataset[()], dtype=dataset.dtype)
            else:
                # Ensure at least 1 row even for very small ratios
                number_of_rows = max(1, int(dataset.shape[0] * ratio))
                # Cap chunk size to the output row count so it never exceeds the data shape
                chunks = (min(dataset.chunks[0], number_of_rows),) if dataset.chunks else None
                output_file.create_dataset(name, data=dataset[:number_of_rows], dtype=dataset.dtype,
                                           compression=dataset.compression,
                                           compression_opts=dataset.compression_opts,
                                           chunks=chunks)
                print(f"  {name}: {dataset.shape[0]} -> {number_of_rows} rows")

    print(f"Written {os.path.getsize(output_path) / 1024**2:.1f} MB to {output_path}")


if __name__ == "__main__":
    main()
