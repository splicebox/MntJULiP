from pathlib import Path
import logging
import argparse
import time
import os
import sys
import shutil
import textwrap


logging.basicConfig(format='mnt-JULiP: %(asctime)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


import pandas as pd
import numpy as np

from utils import get_bam_file_dataframe, get_conditions
from utils import generate_splice_files, get_splice_file_dataframe
from utils import process_annotation, process_introns_with_annotation
from utils import process_introns
from utils import write_pred_intron_file, write_diff_nb_intron_file, write_diff_dm_intron_file, write_diff_dm_group_file
from models import NB_model, DM_model


def get_arguments():
    version = 'MntJULiP v1.1.0'
    parser = argparse.ArgumentParser(description=version, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version=version)
    parser.add_argument('--bam-list', type=str, help='a text file that contains the list of the BAM file paths and sample conditions.')
    parser.add_argument('--splice-list', type=str, help='a text file that contains the list of the SPLICE file paths and sample conditions.')
    parser.add_argument('--anno-file', type=str, default='', help='annotation file in GTF format.')
    parser.add_argument('--no-save-tmp', action='store_false', default=True,
                        help='by default the splice files are saved in order to save time when rerun the program on the same dataset.')
    parser.add_argument('--out-dir', type=str, default='./out', help='output folder to store the results and temporary files. (default: ./out)')
    parser.add_argument('--num-threads', type=int, default=4, help='number of CPU cores use to run the program. (default: 4)')
    parser.add_argument('--min-count', type=int, default=5,
                        help='average intron read count for all conditions that less than this value will be considered as low data and will be skipped (default: 5).')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='size of batch of intron/groups feed to the models for each process (default: 1000).')
    parser.add_argument('--error-rate', type=float, default=0.05, help='family-wise error rate for FDR (default: 0.05).')
    parser.add_argument('--group-filter', type=float, default=1., help='minimum read counts required for group in at least one sample (default: 1).')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode for showing more information.')
    parser.add_argument('--aggressive-mode', action='store_true', default=False,
                        help='set MnutJULiP to aggressive mode for highly dispersed data (e.g. cancer data)')
    parser.add_argument('--method', type=str, default='fdr_bh',
                        help=textwrap.dedent('''\
    method used for testing and adjustment of p-values (default: 'fdr_bh')
    - `bonferroni` : one-step correction
    - `sidak` : one-step correction
    - `holm-sidak` : step down method using Sidak adjustments
    - `holm` : step-down method using Bonferroni adjustments
    - `simes-hochberg` : step-up method  (independent)
    - `hommel` : closed method based on Simes tests (non-negative)
    - `fdr_bh` : Benjamini/Hochberg  (non-negative)
    - `fdr_by` : Benjamini/Yekutieli (negative)
    - `fdr_tsbh` : two stage fdr correction (non-negative)
    - `fdr_tsbky` : two stage fdr correction (non-negative)
    '''))

    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


def main():
    args = get_arguments()

    # logging.info('Creating output folder ...')
    out_dir = Path(args.out_dir)
    out_data_dir = out_dir / 'data'
    out_data_dir.mkdir(parents=True, exist_ok=True)
    num_threads = args.num_threads
    save_tmp = args.no_save_tmp
    error_rate = args.error_rate
    method = args.method
    anno_file = args.anno_file
    count = args.min_count
    batch_size = args.batch_size
    debug = args.debug
    group_filter = args.group_filter
    aggressive_mode = args.aggressive_mode

    if args.bam_list:
        file_list_df = get_bam_file_dataframe(args.bam_list)
        work_dir = os.path.dirname(os.path.abspath(__file__))
        logging.info('Generating splice files (or reusing splice files if save-tmp set to true and splice files exist) ...')
        generate_splice_files(work_dir, out_data_dir, file_list_df, num_threads, save_tmp=save_tmp)
    elif args.splice_list:
        logging.info('Reading splice files ...')
        file_list_df = get_splice_file_dataframe(args.splice_list, out_data_dir)
    else:
        raise Exception('Please provide a BAM file list or SPLICE file list!')

    conditions, labels = get_conditions(file_list_df)
    num_samples = conditions.shape[0]

    logging.info(f"Processing {num_samples} samples ...")
    anno_info = None
    if anno_file:
        anno_intron_dict, start_site_genes_dict, end_site_genes_dict = process_annotation(anno_file)
        df, index_df, anno_intron_dict = process_introns_with_annotation(out_data_dir, num_samples, anno_intron_dict, start_site_genes_dict,
                                                                         end_site_genes_dict, conditions, num_threads=num_threads)
        anno_info = (anno_intron_dict, start_site_genes_dict, end_site_genes_dict)
    else:
        df, index_df = process_introns(out_data_dir, num_samples, num_threads=num_threads)

    logging.info(f'{df.shape[0]} candidate introns found.')
    df['label'] = pd.Series(np.zeros(df.shape[0]).astype(int), index=df.index)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = Path(base_dir) / 'lib'

    logging.info('Fitting Negative Binomial models ...')
    start_time = time.time()
    diff_nb_intron_dict, pred_intron_dict = NB_model(df, conditions, model_dir,
                                             num_workers=num_threads, count=count, error_rate=error_rate,
                                             method=method, batch_size=batch_size, aggressive_mode=aggressive_mode)
    logging.info(f'Finished! Took {time.time() - start_time:0.2f} seconds.')

    logging.info('Fitting Dirichlet Multinomial models ...')
    start_time = time.time()
    diff_dm_intron_dict, diff_dm_group_dict = DM_model(df, index_df, conditions, model_dir,
                                                        num_workers=num_threads, error_rate=error_rate,
                                                        method=method, batch_size=batch_size, group_filter=group_filter,
                                                        aggressive_mode=aggressive_mode)
    logging.info(f'Finished! Took {time.time() - start_time:0.2f} seconds.')

    logging.info('Writing results ...')
    write_pred_intron_file(df, conditions, labels, pred_intron_dict, out_dir, anno_info)
    write_diff_nb_intron_file(labels, diff_nb_intron_dict, out_dir, anno_info, debug=debug)
    write_diff_dm_intron_file(labels, diff_dm_intron_dict, out_dir, anno_info)
    write_diff_dm_group_file(diff_dm_group_dict, out_dir, anno_info)

    if not save_tmp:
        shutil.rmtree(out_data_dir)

    logging.info('Done!')


if __name__ == "__main__":
  main()
