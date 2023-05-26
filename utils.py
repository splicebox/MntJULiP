import os, gzip, subprocess
from pathlib import Path
from functools import reduce
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from dask import delayed, compute
import dask.dataframe as dd
import os
import glob
import gzip


def get_bam_file_dataframe(list_file):
    return pd.read_csv(list_file, sep='\t')


def get_splice_file_dataframe(list_file, data_dir):
    splice_file_df = pd.read_csv(list_file, sep='\t')
    for file in os.listdir(data_dir):
        if file.endswith(".splice") or file.endswith(".splice.gz"):
            os.remove(data_dir / file)
    for i, file in enumerate(splice_file_df['sample']):
        if os.path.isfile(file) and os.path.getsize(file) > 0:
            if file.endswith(".gz"):
                os.symlink(file, data_dir / f"sample_{i+1}.splice.gz")
            else:
                os.symlink(file, data_dir / f"sample_{i+1}.splice")
        elif not os.path.isfile(file):
            raise FileNotFoundError(f'{file} does not exist!')
        else:
            raise Exception(f'{file} is empty!')
    return splice_file_df


def get_conditions(file_df):
    conditions = pd.get_dummies(file_df['condition'])
    labels = conditions.columns.values.tolist()
    return conditions.values, labels


def generate_splice_data(work_dir, out_dir, filename, bam_file, save_tmp):
    splice_file = out_dir / f'{filename}.splice.gz'
    if save_tmp and splice_file.exists():
        return
    else:
        if splice_file.exists():
            splice_file.unlink()
        # command = f'bin/junc {bam_file} --nh 5'
        command = [f'{work_dir}/bin/junc', bam_file]
        with gzip.open(splice_file, 'wb') as f:
            f.write(subprocess.Popen(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT).stdout.read())


def generate_splice_files(work_dir, out_dir, bam_file_df, num_threads=4, save_tmp=True):
    delayed_results = []
    for i, bam_file in enumerate(bam_file_df['sample']):
        delayed_results.append(delayed(generate_splice_data)(work_dir, out_dir, f'sample_{i+1}', bam_file, save_tmp))
    _ = compute(*delayed_results, traverse=False, num_workers=num_threads)


def process_introns(data_dir, num_samples, num_threads=4):
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    dfs = []
    dtype_dict = {'chromosome': 'string'}
    for i in range(num_samples):
        columns = ["chromosome", "start", "end", f"{i+1}_count", "strand"]
        if os.path.exists(data_dir / f'sample_{i+1}.splice.gz'):
            filename = data_dir / f'sample_{i+1}.splice.gz'
            _df = dd.read_csv(filename, sep=' ', blocksize=None,
                        names=columns, usecols=[0, 1, 2, 3, 4], compression='gzip', dtype=dtype_dict)
        elif os.path.exists(data_dir / f'sample_{i+1}.splice'):
            filename = data_dir / f'sample_{i+1}.splice'
            _df = dd.read_csv(filename, sep=' ', blocksize=None,
                        names=columns, usecols=[0, 1, 2, 3, 4], dtype=dtype_dict)
        else:
            raise Exception("Splice file doesn't exist!")

        # drop the negative read counts if any
        _df = _df[_df[f"{i+1}_count"] >= 0]
        dfs.append(_df)

    while len(dfs) > 1:
        _list = []
        for chunk in chunks(dfs, 5):
            df = delayed(reduce)(lambda x, y: dd.merge(x, y, how='outer', on=['chromosome', 'start', 'end', 'strand']), chunk)
            _list.append(df)
        dfs = _list

    df = compute(*dfs, num_workers=num_threads)[0]
    df.fillna(0, inplace=True)

    if num_samples > 10:
        column_names = list(set(df.columns.values) - set(['chromosome', 'start', 'end', 'strand']))
        df = df[(df[column_names] > 3).any(axis=1)]

    coord_columns = ['chromosome', 'strand', 'start', 'end']
    index_df = df[coord_columns].copy()
    index_df['index'] = df[coord_columns].apply(lambda x: tuple(x), axis=1)
    index_df.set_index(coord_columns, inplace=True)

    df['index'] = df[coord_columns].apply(lambda x: tuple(x), axis=1)
    df.drop(coord_columns, axis=1, inplace=True)
    df.set_index('index', inplace=True)

    return df, index_df


def process_introns_with_annotation(data_dir, num_samples, anno_intron_dict, start_site_genes_dict,
                                    end_site_genes_dict, conditions, num_threads=4):
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def to_set(x):
        return set(x)

    def to_list(x):
        return list(x)

    # filter to remove the introns with spilce sites in different genes
    def _filter(introns, start_site_genes_dict, end_site_genes_dict):
        booleans =[]
        for (_chr, strand, start, end) in introns:
            start_genes = set()
            if (_chr, strand, start) in start_site_genes_dict:
                start_genes = start_site_genes_dict[(_chr, strand, start)]

            end_genes = set()
            if (_chr, strand, end) in end_site_genes_dict:
                end_genes = end_site_genes_dict[(_chr, strand, end)]

            if not(start_genes.intersection(end_genes)) and start_genes and end_genes:
                booleans.append(False)
            else:
                booleans.append(True)

        return booleans

    dfs = []
    dtype_dict = {'chromosome': 'string'}
    for i in range(num_samples):
        columns = ["chromosome", "start", "end", f"{i+1}_count", "strand"]
        if os.path.exists(data_dir / f'sample_{i+1}.splice.gz'):
            filename = data_dir / f'sample_{i+1}.splice.gz'
            _df = dd.read_csv(filename, sep=' ', blocksize=None,
                        names=columns, usecols=[0, 1, 2, 3, 4], compression='gzip', dtype=dtype_dict)
        elif os.path.exists(data_dir / f'sample_{i+1}.splice'):
            filename = data_dir / f'sample_{i+1}.splice'
            _df = dd.read_csv(filename, sep=' ', blocksize=None,
                        names=columns, usecols=[0, 1, 2, 3, 4], dtype=dtype_dict)
        else:
            raise Exception("Splice file doesn't exist!")

        # drop the negative read counts if any
        _df = _df[_df[f"{i+1}_count"] >= 0]
        dfs.append(_df)

    while len(dfs) > 1:
        _list = []
        for chunk in chunks(dfs, 5):
            df = delayed(reduce)(lambda x, y: dd.merge(x, y, how='outer', on=['chromosome', 'start', 'end', 'strand']), chunk)
            _list.append(df)
        dfs = _list

    df = compute(*dfs, num_workers=num_threads)[0]
    df.fillna(0, inplace=True)

    # filter out very low expressed intron candidates
    if num_samples > 10:
        column_names = [f"{i+1}_count" for i in range(num_samples)]
        df = df[(df[column_names] > 3).any(axis=1)]

    ## create index for df and index_df for later usage
    coord_columns = ['chromosome', 'strand', 'start', 'end']
    df['index'] = df[coord_columns].apply(lambda x: tuple(x), axis=1)
    booleans = _filter(df['index'].tolist(), start_site_genes_dict, end_site_genes_dict)
    df = df[booleans]

    index_df = df[coord_columns].copy()
    index_df['index'] = df['index'].copy()
    index_df.set_index(coord_columns, inplace=True)

    df.drop(coord_columns, axis=1, inplace=True)
    df.set_index('index', inplace=True)

    ## create intron-gene data frame
    _list = []
    for coord, genes in anno_intron_dict.items():
        for gene in genes:
            _list.append((coord, gene))

    anno_intron_df = pd.DataFrame.from_records(_list, columns=['index', 'gene'])
    anno_intron_df = anno_intron_df.loc[anno_intron_df['index'].isin(df.index)]

    gene_introns_dict = anno_intron_df.groupby('gene').agg({'index': to_set})['index'].to_dict()
    intron_genes_dict = anno_intron_df[anno_intron_df.duplicated(['index'], keep=False)].groupby('index').agg({'gene': to_list})['gene'].to_dict()

    gene_set = set()
    for intron, genes in intron_genes_dict.items():
        if len(genes) > 1:
            for i in range(0, len(genes) - 1):
                for j in range(i + 1, len(genes)):
                    iset1 = gene_introns_dict[genes[i]]
                    iset2 = gene_introns_dict[genes[j]]
                    if iset1.issubset(iset2):
                        gene_set.add(genes[i])
                    if iset2.issubset(iset1):
                        gene_set.add(genes[j])


    anno_intron_df = anno_intron_df.loc[~anno_intron_df['gene'].isin(gene_set)]
    anno_intron_dict = anno_intron_df.groupby('index').agg({'gene': to_list})['gene'].to_dict()

    return df, index_df, anno_intron_dict


def _get_gene_names(anno_info, coord):
    (anno_intron_dict, start_site_genes_dict, end_site_genes_dict) = anno_info
    gene_names = set()
    if coord in anno_intron_dict:
        gene_names.update(anno_intron_dict[coord])
    else:
        _chr, strand, start, end = coord
        if (_chr, strand, start) in start_site_genes_dict:
            gene_names.update(start_site_genes_dict[(_chr, strand, start)])
        if (_chr, strand, end) in end_site_genes_dict:
            gene_names.update(end_site_genes_dict[(_chr, strand, end)])
    return gene_names


def get_gene_names(anno_info, coord):
    _chr, strand, start, end = coord
    if strand == '?':
        gene_names = _get_gene_names(anno_info, (_chr, "+", start, end))
        gene_names.update(_get_gene_names(anno_info, (_chr, "-", start, end)))
        return ','.join(gene_names) if gene_names else '.'
    gene_names = _get_gene_names(anno_info, coord)
    return ','.join(gene_names) if gene_names else '.'


def write_pred_intron_file(df, conditions, labels, pred_intron_dict, out_dir, anno_info=None):
    file = out_dir / 'intron_data.txt'
    _list = ['chrom', 'start', 'end', 'strand', 'gene_name', 'status']
    indices = []
    for i, label in enumerate(labels):
        indices.append(np.where(conditions[:, i] > 0)[0])
        _list.append(f"read_counts({label})")

    with open(file, 'w') as f:
        f.write('\t'.join(_list) + '\n')
        rows = df.values.tolist()
        coordinates = df.index.tolist()
        for i in range(len(rows)):
            row_list = rows[i]
            coord = coordinates[i]
            gene_names = get_gene_names(anno_info, coord) if anno_info else '.'
            _chr, strand, start, end = coord
            _list = [_chr, str(start), str(end), strand, gene_names, pred_intron_dict[coord]]
            y = np.array(row_list[:-1], dtype=np.int)
            _list += [','.join(np.take(y, i).astype(str).tolist()) for i in indices]
            f.write('\t'.join(_list) + '\n')


def write_diff_nb_intron_file(labels, diff_nb_intron_dict, out_dir, anno_info=None, debug=False):
    file = out_dir / 'diff_introns.txt'
    _list = ['chrom', 'start', 'end', 'strand', 'gene_name', 'status', 'llr', 'p_value', 'q_value']
    for label in labels:
        _list.append(f"avg_read_counts({label})")
    if debug:
        for label in labels:
            _list.append(f"variance({label})")

    with open(file, 'w') as f:
        f.write('\t'.join(_list) + '\n')
        for coord, value in diff_nb_intron_dict.items():
            p_value, log_likelihood, mus, sigmas, q_value = value
            str_p_value = f"{p_value:.6g}" if p_value is not None else 'NA'
            str_q_value = f"{q_value:.6g}" if q_value is not None else 'NA'
            str_log_likelihood = f"{log_likelihood:.6g}" if log_likelihood is not None else 'NA'
            status = 'TEST' if p_value is not None else 'NO_TEST'
            gene_names = get_gene_names(anno_info, coord) if anno_info else '.'
            _chr, strand, start, end = coord
            _list = [_chr, str(start), str(end), strand, gene_names, status, str_log_likelihood, str_p_value, str_q_value]
            _list += [f"{mu:.2f}" for mu in mus]
            if debug:
                if sigmas is not None:
                    _list += [f"{sigma:.2f}" for sigma in sigmas]
                else:
                    _list += ["NA"] * len(labels)
            f.write('\t'.join(_list) + '\n')


def write_diff_dm_intron_file(labels, diff_dm_intron_dict, out_dir, anno_info=None):
    file = out_dir / 'diff_spliced_introns.txt'
    _list = ['group_id', 'chrom', 'start', 'end', 'strand', 'gene_name']
    for label in labels:
        _list.append(f"psi({label})")

    if len(labels) == 2:
        _list.append("delta_psi")

    with open(file, 'w') as f:
        f.write('\t'.join(_list) + '\n')
        for coord, values in diff_dm_intron_dict.items():
            for value in values:
                group_id, psis = value
                gene_names = get_gene_names(anno_info, coord) if anno_info else '.'
                _chr, strand, start, end = coord
                _list = [group_id, _chr, str(start), str(end), strand, gene_names] + [f"{p:.6g}" for p in psis]

                if len(labels) == 2:
                    dpsi = psis[1] - psis[0]
                    _list += [f"{dpsi:.6g}"]
                f.write('\t'.join(_list) + '\n')


def get_group_gene_names(group, intron_coords, anno_info):
    (anno_intron_dict, start_site_genes_dict, end_site_genes_dict) = anno_info
    gene_names_list = []
    for coord in intron_coords:
        _chr, strand, start, end = coord
        if strand == '?':
            gene_names = _get_gene_names(anno_info, (_chr, "+", start, end))
            gene_names.update(_get_gene_names(anno_info, (_chr, "-", start, end)))
        else:
            gene_names = _get_gene_names(anno_info, coord)
        gene_names_list.append(gene_names)

    gene_names = set.intersection(*gene_names_list)
    if not gene_names:
        gene_names = set.union(*gene_names_list)

    return ','.join(gene_names) if gene_names else '.'


def write_diff_dm_group_file(diff_dm_group_dict, out_dir, anno_info=None):
    file = out_dir / 'diff_spliced_groups.txt'
    _list = ['group_id', 'chrom', 'loc', 'strand', 'gene_name', 'structure', 'llr', 'p_value', 'q_value']

    with open(file, 'w') as f:
        f.write('\t'.join(_list) + '\n')
        for group, value in diff_dm_group_dict.items():
            _chr, strand, loc, structure = group
            group_id, p_value, log_likelihood, intron_coords, q_value = value
            gene_names = get_group_gene_names(group, intron_coords, anno_info) if anno_info else '.'
            _list = [group_id, _chr, str(loc), strand, gene_names, structure,
                    f"{log_likelihood:.6g}", f"{p_value:.6g}", f"{q_value:.6g}"]
            f.write('\t'.join(_list) + '\n')


def get_intron_coords(exon_coords):
    if len(exon_coords) <= 1:
        return []
    else:
        introns = []
        _list = sorted(exon_coords)
        start = _list[0][1]
        for i in range(1, len(_list)):
            end = _list[i][0]
            introns.append((start, end))
            start = _list[i][1]
        return introns


def process_annotation(gtf_file):
    def process_string(_str):
        result_dict = {}
        for substr in _str.strip().split(';'):
            if substr:
                try:
                    key, value = substr.strip().split(' ', 1)
                except ValueError:
                    print(substr)
                result_dict[key] = value.strip('\"')
        return result_dict

    tran_exons_dict = defaultdict(list)
    tran_gene_dict = {}
    start_site_genes_dict = defaultdict(set)
    end_site_genes_dict = defaultdict(set)

    f = gzip.open(gtf_file, 'rt', encoding='utf-8') if gtf_file.endswith(".gz") else open(gtf_file, 'r')
    for line in f:
        if line != '\n' and not(line.startswith('#')):
            items = line.strip().split('\t')
            if items[2] == 'exon':
                _chr, strand = items[0], items[6]
                start, end = int(items[3]), int(items[4])
                info_dict = process_string(items[8])
                tran_id = info_dict['transcript_id']
                tran_exons_dict[(_chr, strand, tran_id)].append((start, end))
                gene_id = info_dict['gene_id']
                # prefer choose gene name as id if available
                if 'gene_name' in info_dict:
                    gene_id = info_dict['gene_name']
                tran_gene_dict[(_chr, strand, tran_id)] = gene_id
                start_site_genes_dict[(_chr, strand, end)].add(gene_id)
                end_site_genes_dict[(_chr, strand, start)].add(gene_id)
    f.close()

    anno_intron_dict = defaultdict(set)
    for key, exon_coords in tran_exons_dict.items():
        _chr, strand, tran_id = key
        introns = get_intron_coords(exon_coords)
        gene_id = tran_gene_dict[(_chr, strand, tran_id)]
        for coord in introns:
            start, end = coord
            anno_intron_dict[(_chr, strand, start, end)].add(gene_id)

    return anno_intron_dict, start_site_genes_dict, end_site_genes_dict

