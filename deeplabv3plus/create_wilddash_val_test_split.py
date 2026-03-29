"""
Create WildDash val (tuning) / test split from wilddash_val_list.txt.
Use 30% for validation tuning, 70% for final test. Seed 42 for reproducibility.
"""
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", type=str, default="wilddash_val_list.txt", help="Full image list")
    parser.add_argument("--val-out", type=str, default="wilddash_val_tune.txt", help="Validation (tuning) list output")
    parser.add_argument("--test-out", type=str, default="wilddash_test.txt", help="Test list output")
    parser.add_argument("--val-frac", type=float, default=0.3, help="Fraction for val (default 0.3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.list, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    import random
    random.seed(args.seed)
    random.shuffle(lines)
    n_val = max(1, int(len(lines) * args.val_frac))
    val_lines = lines[:n_val]
    test_lines = lines[n_val:]
    with open(args.val_out, "w") as f:
        f.write("\n".join(val_lines) + "\n")
    with open(args.test_out, "w") as f:
        f.write("\n".join(test_lines) + "\n")
    print(f"Split: {len(val_lines)} val (tuning), {len(test_lines)} test. Wrote {args.val_out}, {args.test_out}")

if __name__ == "__main__":
    main()
