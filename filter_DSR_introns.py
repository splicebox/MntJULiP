from pathlib import Path
import argparse
import sys


def get_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='A script to filter significant DSR introns')
    parser.add_argument('--dir', type=str, help='directory that contains diff_spliced_introns.txt and diff_spliced_groups.txt')
    parser.add_argument('--pvalue', type=float, default=0.05, help='filter by p-value (default 0.05)')
    parser.add_argument('--qvalue', type=float, default=1, help='filter by q-value (default 1.0)')
    parser.add_argument('--dpsi', type=float, default=0.05, help='filter by absolute value of dPSI (default 0.05)')

    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


def main():
    args = get_arguments()
    base_dir = Path(args.dir)
    file = base_dir / 'diff_spliced_groups.txt'
    with open(file, 'r') as f:
        lines = f.readlines()

    significant_groups = set()
    for line in lines[1:]:
        group_id, _chr, _, strand, gene_names_str, _, _, p_value, q_value = line.strip().split('\t')

        p_value, q_value = float(p_value), float(q_value)
        if p_value <= args.pvalue and q_value <= args.qvalue:

            significant_groups.add(group_id)

    file = base_dir / 'diff_spliced_introns.txt'
    with open(file, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:
        group_id, _chr, start, end, strand, _, _, _, dpsi = line.strip().split('\t')
        start, end, dpsi = int(start), int(end), float(dpsi)
        if abs(dpsi) >= args.dpsi and group_id in significant_groups:
            print(line.strip())


if __name__ == "__main__":
    main()
