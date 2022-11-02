## MntJULiP ##

MntJULiP is a method for comprehensively and accurately quantifying splicing differences at intron level from collections of RNA-seq data. MntJULiP detects both differences in intron abundance levels, herein called differential splicing abundance (DSA), and differences in intron splicing ratios relative to the local gene output, termed differential splicing ratio (DSR). MntJULiP uses a Bayesian mixture model that allows comparison of multiple conditions.  

Described in:

Yang G, Sabunciyan S, and Florea L (2022). Comprehensive and scalable quantification of splicing differences with MntJULiP. *Genome Biol* [23:195](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02767-y). 

Data described in the article can be found [here](http://ccb.jhu.edu/software/MntJULiP/). Supplementary [scripts](https://github.com/splicebox/MntJULiP/blob/master/MntJULiP_scripts.tar.gz) are provided with this distribution. 


```
Copyright (C) 2019-2022, and GNU GPL v3.0, by Guangyu Yang, Liliana Florea
```

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.  

### <a name="table-of-contents"></a> Table of contents
- [What is MntJULiP?](#what-is-mntjulip)
- [Installation](#installation)
- [Usage](#usage)
- [Input/Output](#inputoutput)
- [Visualization](#visualization)
- [Support](#support)

### <a name="what-is-mntjulip"></a> What is MntJULiP?
MntJULiP is a high-performance Python package for comprehensive and accurate quantification of splicing differences from RNA-seq data. It is designed to detect both changes in splicing ratios (*differential splicing ratio*, DSR), and changes in the absolute splicing levels (*differential splicing abundance*, DSA).  MntJULiP works at the level of introns, or splice junctions, and therefore it is assembly-free, and can be used with or without a reference annotation. MntJULiP is fully scalable, and can work with very large RNA-seq data collections, as well as with just a few samples.


MntJULiP takes as input RNA-seq read alignments, such as those generated by Tophat2 or STAR, and uses Bayesian mixture models to detect changes in intron splicing between conditions. Its statistical underpinnings include a Dirichlet multinomial mixture model (DSR test) and a zero-inflated negative binomial mixture model (DSA test), coupled with log likelihood ratio tests. Unlike other tools, MntJULiP can perform multi-way comparisons, which may be desirable for complex and time-series experiments.

#### Salient features  
- Novel statistical models for detecting differential intron abundance (DSA) and differential intron splicing ratio (DSR) across samples;
- Ability to perform multi-way comparisons;  
- Can be used with or without a reference annotation;  
- Multi-threaded and highly scalable.

### <a name="installation"></a> Installation
MntJULiP is written in Python. To download the source code, clone this GitHub repository:

```
git clone https://github.com/splicebox/MntJULiP.git
```

#### System requirements
* Linux or Mac  
* Python 3.7 or later
* gcc 6.4.0 or later

#### Prerequisites
MntJULiP has the following dependencies:
* [PyStan](https://pystan.readthedocs.io/), a package for statistical modeling and high-performance statistical computing.  
* [NumPy](https://numpy.org/), a fundamental package for scientific computing with Python.    
* [SciPy](https://www.scipy.org/), a Python-based package for mathematics, science, and engineering.  
* [Statsmodels](https://www.statsmodels.org/), a Python module for the estimation of different statistical models, conducting statistical tests and data exploration.  
* [Pandas](https://pandas.pydata.org/), a fast, powerful and flexible open source data analysis and manipulation tool.  
* [Dask](https://dask.org/), a Python package that provides advanced parallelism for analytics.  

The required packages can be installed from the [Python Package Index](https://pypi.org/) using pip3:
```
pip3 install --user numpy scipy pandas "pystan<3" statsmodels "dask[complete]"
```

Alternatively, run setup.py to install MntJULiP and all the required packages; make sure you have the correct versions of Python and gcc in your path, and that they are loaded in that order. 
```
cd MntJULiP; python setup.py install
```
Add "--user" option if you do not have the root/administrative privileges:
```
cd MntJULiP; python setup.py install --user
```
If you encounter "error: can't combine user with prefix, exec_prefix/home, or install_(plat)base", try:
```
python setup.py install --user --prefix=
```
For some linux distributions, "python" command usally links to python 2.x by default. If you encounter errors related to the python packages, please try to replace "python" with "python3", for example, use:
```
python3 setup.py install
```
For [Conda](https://docs.conda.io/) user, it's easier to install MntJULiP without manually resolving the depedency issues.   
Create the mntjulip environment by
```
conda create -n mntjulip-env python=3.9
```
Activate mntjulip-env by
```
conda activate mntjulip-env
```
Then run the above python commands to install MntJULiP

### <a name="usage"></a> Usage
```
Usage: python run.py [options] [--bam-list bam_file_list.txt | --splice-list splice_file_list.txt]

required arguments:
  --bam-list BAM_LIST   a text file that contains the list of the BAM file
                        paths and sample conditions
  OR
  --splice-list SPLICE_LIST
                        a text file that contains the list of the SPLICE file
                        paths and sample conditions
optional arguments:
  --anno-file ANNO_FILE
                        annotation file in GTF format
  --out-dir OUT_DIR     output folder to store the results and temporary files (default: ./out)
  --num-threads NUM_THREADS
                        number of CPU cores use to run the program (default: 4)
  -v, --version         show program's version number and exit
  -h, --help            show this help message and exit
```

Here is an example on how to run MntJULiP with the alignment files and the GENCODE annotation:
```
ANNO_FILE="gencode.v22.annotation.gtf"
BAM_LIST="bam_file_list.txt"
python run.py --bam-list ${BAM_LIST} \
               --anno-file ${ANNO_FILE} \
               --num-threads 8           
```

The 'bam_file_list' is a .txt file with two columns separated by 'tab' or '\t'. Here is an example:
```
sample   condition
sample1.bam    control
sample2.bam    case
```

Note that in the current version MntJULiP automatically determines a reference condition, by sorting the conditions and choosing the lexicographically smallest one.

Extracting the splice junctions and their read counts needed for quantification from the .bam files is the first and most time consuming step. It may be beneficial to avoid recalculating the values on subsequent runs of the data or subgroups. Also, in some cases the .bam files may not be available, or the splice junctions can be obtained from other sources. For efficient processing and to accommodate such cases, MntJULiP can work directly with the .splice files, instead of the .bam files, as input.

Here is an example on how to run MntJULiP with the GENCODE annotation and the splice file list:
```
SPLICE_LIST="splice_file_list.txt"
ANNO_FILE="gencode.v22.annotation.gtf"
python run.py --splice-list ${SPLICE_LIST} \
               --anno-file ${ANNO_FILE} \
               --num-threads 8 
```
The 'splice_file_list.txt' is a .txt file with two columns separated by 'tab' or '\t'. Here is an example:
```
sample   condition
sample1.splice    control
sample2.splice    case
```
The .splice file for a given sample is a space or ' ' separated file with at least 5 columns "chrom start end count strand" (the header is excluded):
```
chr1 1311924 1312018 100 -
```
A splice file may have additional columns; for instance, those generated by the *junc* tool included with this package will distinguish the numbers of uniquely and multiply mapped supporting reads:
```
chr1 1311924 1312018 100 - 67 33 ...
```

### <a name="inputoutput"></a> Input/Output

### Input

The main input of MntJULiP is a list of BAM files containing the RNA-seq read alignments sorted by genomic coordinates. Currently, MntJULiP has been adapted to alignments generated with Tophat2 and STAR with/without '--outSAMstrandField intronMotif' option, for example:
```
STAR --genomeDir ${STARIDX_DIR} \
     --outSAMstrandField intronMotif \
     --readFilesIn ${DATA_DIR}/${name}_1.fastq ${DATA_DIR}/${name}_2.fastq \
     --outSAMtype BAM SortedByCoordinate \
     --outFileNamePrefix ${WORK_DIR}/${name}/
```

Alternatively, MntJULiP can take as input .splice files generated by an external program, such as the junc tool included with this package, or other versions from our [PsiCLASS](https://github.com/splicebox/PsiCLASS) or [CLASS2](https://sourceforge.net/p/splicebox/wiki/Home/) software.
```
junc my_alignment_file.bam -a > my_alignment_file.splice
```

### Output

Output generated by MntJULiP includes four types of files:

*diff_spliced_introns.txt* & *diff_spliced_groups.txt*: contains the results of the differential intron splicing ratio (DSR) analysis;  
*diff_introns.txt*: contains the results of the differential intron abundance (DSA) analysis;  
*intron_data.txt*: contains information about the introns (splice junctions), including genomic coordinates, read counts, average abundance levels, and others.  

These files are described in detail below.

In addition, the directory will contain a temporary folder named 'data', containing the .splice files generated by MntJULiP.


#### 1. DSR analysis: *diff_spliced_introns.txt* & *diff_spliced_groups.txt*

The two files contain information about differential spliced introns determined by the DSR criteria and their group ('bunch') context. A 'bunch' is a group of exons that share the same junction, either the lower coordinate ('i') or the higher coordinate ('o') of the intron(s) along the genomic axis. The *diff_spliced_groups.txt* file lists each 'bunch' along with information about the log-likelihood DSR test and its significance: 

```
group_id        chrom   loc     strand  gene_name       structure       llr     p_value q_value
g000001 chr1    3421702 -       Xkr4    o       1.85645 0.156227        0.477186
```

The file *diff_spliced_introns.txt* lists all introns by group ('bunch') along with their location on the genome, Percent Splice In (PSI) value estimated for each of the conditions in the comparison, and the difference in PSI values *dPSI=PSI2-PSI1*. For multi-way comparisons, the dPSI value is excluded.
```
group_id        chrom   start   end     strand  gene_name       psi(control)    psi(epileptic)  delta_psi
g000001 chr1    3216968 3421702 -       Xkr4    0.843335        0.784751        -0.0585835
g000001 chr1    3323760 3421702 -       Xkr4    0.156665        0.215249        0.0585835
```

To determine introns that are differentially spliced by the DSR test, one may query the two files for introns in 'bunches' that satisfy the statistical significance condition (*e.g.*, *p-val<=0.05* or *q-val<=0.05*) **and** for which the dPSI value exceeds a pre-defined cutoff (*e.g.*, *|dPSI|>=0.05*). For convenience, the script *filter_DSR_introns.py* was provided with this package:

```
filter_DSR_introns.py [-h] [--dir DIR] [--pvalue PVALUE] [--qvalue QVALUE] [--dpsi DPSI]

optional arguments:
  -h, --help       show this help message and exit
  --dir DIR        directory that contains diff_spliced_introns.txt and diff_spliced_groups.txt
  --pvalue PVALUE  filter by p-value (default 0.05)
  --qvalue QVALUE  filter by q-value (default 1.0)
  --dpsi DPSI      filter by absolute value of dPSI (default 0.05)
```

Note that, by virtue of how 'bunches' are constructed, an intron may be reported more than one time, for instance the exon-spanning intron in an exon skipping event will appear in two groups, one for each of its two endpoints. Additionally, more than one exon from the same 'bunch' may satisfy the statistical test and be reported. Our visualization tool [Jutils](https://github.com/splicebox/Jutils/) selects an unbiased set of  introns for display in heatmaps, following the criteria outlined [here](https://github.com/splicebox/Jutils/blob/master/notes/aggregates.md).

#### 2. DSA analysis: *diff_introns.txt*

The file *diff_introns.txt* contains information about differential spliced introns determined by the DSA criteria. It lists all MntJULiP-selected introns, individually, along with their location on the genome, abundance (expression) estimates in terms of normalized read counts in each condition, and information about the log-likelihood ratio test and its significance: 

```
chrom   start   end     strand  gene_name       status  llr     p_value q_value avg_read_counts(control)        avg_read_counts(epileptic)
chr1    3207317 3213439 -       Xkr4    TEST    0.718835        0.696729        0.89138 59.64   41.36
```

Introns that are differentially spliced by the DSA test can be determined directly from the file by querying the entries for statistical significance and/or other criteria. 


#### 3. Supporting intron data

The file *intron_data.txt* contains supporting data for all introns detected by MntJULiP that passed the initial internal quality filters.

```
chrom   start   end     strand  gene_name       status  read_counts(control)    read_counts(epileptic)
chr1    3207317 3213439 -       Xkr4    OK      76,26,66,51,45,62,22,8,96,60,105,8,60,89,71,77,79,47,29,16,52,98,86,23  26,61,44,34,92,51,55,24,51,25,23,20,89,29,19,39,97,51,5,43
chr1    3207317 3213609 -       Xkr4    LO_DATA 1,0,2,1,1,2,1,2,4,3,2,0,4,3,5,2,1,0,1,0,3,2,3,1 1,0,0,3,1,2,1,0,1,1,3,0,3,1,0,1,0,0,1,3
```

It lists all introns individually along with their genomic location, status of the test (*OK* or *LO_DATA*), and the raw read counts for the intron in all samples, grouped by condition. This information can be used, for instance, to further filter introns with low support across all or subsets of the samples, or from genes with low expression levels.


### <a name="visualization"></a> Visualization

The output generated by MntJULiP DSR and DSA can be further visualized using the [Jutils](https://github.com/splicebox/Jutils/) visualization suite. Jutils can generate heatmaps of differentially spliced events, sashimi plots, and Venn diagrams of gene sets determined to be differentially spliced based on program or user-defined criteria, using information extracted directly from the MntJULiP output.


### <a name="support"></a> Support
Contact: etzlanim@gmail.com, florea@jhu.edu, or submit an [Issue](https://github.com/splicebox/MntJULiP/issues) through this Github page.   

#### License information
See the file LICENSE for information on the history of this software, terms
& conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.
