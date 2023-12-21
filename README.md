## MntJULiP ##

MntJULiP is a program for comprehensively and accurately quantifying splicing differences at intron level from collections of RNA-seq data. MntJULiP detects both differences in intron abundance levels, or differential splicing abundance (DSA), and differences in intron splicing ratios relative to the local gene output, or differential splicing ratio (DSR). MntJULiP uses a Bayesian mixture model that allows comparison of multiple conditions, and can model the effects of covariates to enable analyses of large and complex human data from disease cohorts and population studies. 

Described in:

- Yang G, Sabunciyan S, and Florea L (2022). Comprehensive and scalable quantification of splicing differences with MntJULiP. *Genome Biol* [23:195](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02767-y). [[Suppl. data](http://ccb.jhu.edu/software/MntJULiP/), [Suppl. scripts](https://github.com/splicebox/MntJULiP/blob/master/MntJULiP_scripts.tar.gz)]
- Lui WW, Yang G, and Florea L (2023). MntJULiP and Jutils: Differential splicing analysis of RNA-seq data with covariates. *Submitted.*


For the original version of MntJULiP, please refer to the [master](https://github.com/splicebox/MntJULiP/tree/master) branch.

```
Copyright (C) 2019-2023, and GNU GPL v3.0, by †Guangyu Yang, †Wui Wang Lui, Liliana Florea († equal contributors)
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
MntJULiP is a high-performance Python package for comprehensive and accurate quantification of splicing differences from RNA-seq data. It uses Bayesian mixture models to detect changes in splicing ratios (*differential splicing ratio*, DSR) and in absolute splicing levels (*differential splicing abundance*, DSA). Its statistical underpinnings include a Dirichlet multinomial mixture model, to test for differences in the splicing ratio, and a zero-inflated negative binomial mixture model, to test for differential splicing abundance. MntJULiP works at the level of introns, or splice junctions, and therefore it is assembly-free, and can be used with or without a reference annotation. MntJULiP can perform multi-way comparisons, which may be desirable for complex and time-series experiments. Additionally, it can model confounders such as age, sex, BMI and others, and removes their biases from the data to allow for accurate comparisons. MntJULiP is fully scalable, and can work with data sets from a few to hundred or thousands of RNA-seq samples. 

MntJULiP was designed to be compatible with alignments generated by [STAR](https://github.com/alexdobin/STAR) or [Tophat2](https://github.com/DaehwanKimLab/tophat2), but may work with output from other aligners. 

#### Features  
- A novel assembly-free model for detecting differential intron abundance and differential intron splicing ratio across samples;
- Allows multi-way comparisons;  
- Incorporates the treatment of covariates within the models;
- Can be used with or without a reference annotation;  
- Multi-threaded and highly scalable, can process hundreds of samples in hours.

### <a name="installation"></a> Installation
MntJULiP is written in Python, you can install the latest version from our GitHub repository. To download the code, you can clone this repository by

```
git clone https://github.com/splicebox/MntJULiP.git
```

#### System requirement
* Linux or Mac   
* Python 3.7 or later
* gcc 6.4.0 or later

#### Prerequisites:
MntJULiP has the following dependencies:
* [PyStan](https://pystan.readthedocs.io/), a package for statistical modeling and high-performance statistical computation.  
* [NumPy](https://numpy.org/), a fundamental package for scientific computing with Python.    
* [SciPy](https://www.scipy.org/), a Python-based package for mathematics, science, and engineering.  
* [Statsmodels](https://www.statsmodels.org/), a Python module for the estimation of different statistical models, conducting statistical tests and data exploration.  
* [Pandas](https://pandas.pydata.org/), a fast, powerful, flexible and easy to use open source data analysis and manipulation tool.  
* [Dask](https://dask.org/), a Python package that provides advanced parallelism for analytics.  
* [scikit-learn](https://scikit-learn.org/), a Python simple and efficient tools for predictive data analysis.

The required packages may be installed using conda:
```
cd MnJULiP
conda env create -f mntjulip2.yml
#  run setup.py to install MntJULiP and all the required packages
module load cmake
python3 setup.py install  
```

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
  --raw-counts-only     output sample-level raw values only (default: both raw and estimated values)
  -v, --version         show program's version number and exit
  -h, --help            show this help message and exit
```

Here is an example to run MntJULiP with a set of alignment files and the GENCODE annotation:
```
ANNO_FILE="gencode.v22.annotation.gtf"
BAM_LIST="bam_file_list.txt"
python3 run.py --bam-list ${BAM_LIST} \
               --anno-file ${ANNO_FILE} \
               --threads 8            
```

The 'bam_list' is a .txt file with columns separated by tabs ('\t'); covariate columns are optional. Here is an example:
```
sample	condition	covariate1	covariate2
sample1.bam	control	Male	18
sample2.bam	case	Female	49
```

Note that in the current version, MntJULiP automatically determines a reference condition, by sorting the conditions and choosing the lexicographically smallest one.

Extracting the splice junctions and their read counts needed for quantification from the .bam files is the first and most time consuming step. It may be beneficial to avoid recalculating the values on subsequent runs of the data or subgroups. Also, in some cases the .bam files may not be available, or the splice junctions can be obtained from other sources. For efficient processing and to accommodate such cases, MntJULiP can work directly with the .splice files, instead of the .bam files, as input.

Here is an example of how to run MntJULiP with the GENCODE annotation and the splice file list:
```
SPLICE_LIST="splice_file_list.txt"
ANNO_FILE="gencode.v22.annotation.gtf.gz"
python run.py --splice-list ${SPLICE_LIST} \
               --anno-file ${ANNO_FILE} \
               --num-threads 8 
```
The 'splice_file_list.txt' is a .txt file with columns separated by 'tab' or '\t'; covariate columns are optional. Here is an example:
```
sample  condition	covariate1	covariate2
sample1.splice.gz  control	Male	18
sample2.splice.gz  case	Female	49
```
The .splice file for a given sample is a space or ' ' separated file with at least 5 columns "chrom start end count strand" (the header is excluded):
```
chr1 1311924 1312018 100 -
```
A splice file may have additional columns; for instance, those generated by the *junc* tool included in this package will distinguish the numbers of uniquely and multimapped supporting reads:
```
chr1 1311924 1312018 100 - 67 33 ...
```

### <a name="inputoutput"></a> Input/Output

### Input
The main _input_ of MntJULiP is a list of BAM files containing RNA-seq read alignments. The BAM files can be generated by [STAR](https://github.com/alexdobin/STAR) with or without '--outSAMstrandField intronMotif' option.
```
STAR --genomeDir ${STARIDX_DIR} \
     --outSAMstrandField intronMotif \
     --readFilesIn ${DATA_DIR}/${name}_1.fastq ${DATA_DIR}/${name}_2.fastq \
     --outSAMtype BAM SortedByCoordinate \
     --outFileNamePrefix ${WORK_DIR}/${name}/
```
Alternatively, MntJULiP can take as input .splice files generated by an external program, such as the junc tool included in this package, or other versions from our [PsiCLASS](https://github.com/splicebox/PsiCLASS) or [CLASS2](http://sourceforge.net/projects/splicebox) packages. 
```
junc my_alignment_fil.bam -a > my_alignment_file.splice
```

### Output
Output generated by MntJULiP includes five types of files:

- *diff_spliced_introns.txt* and *diff_spliced_groups.txt*: contains the results of the differential intron splicing ratio (DSR) analysis;  
- *diff_introns.txt*: contains the results of the differential intron abundance (DSA) analysis;  
- *intron_data.txt* and *group_data.txt*: contains information about the introns (splice junctions), including genomic coordinates, raw and estimated read counts, average abundance levels, etc.; and respectively groups, including genomic location, splicing ratios (*PSI* values) both calculated from read counts and estimated by the model, and others.  

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


#### 3. Supporting intron & group data

The file *intron_data.txt* contains supporting data for all introns detected by MntJULiP that passed the initial internal quality filters.
```
chrom   start   end     strand  gene_name       status  read_counts(control)    read_counts(epileptic)
chr1    3207317 3213439 -       Xkr4    OK      76,26,66,51,45,62,22,8,96,60,105,8,60,89,71,77,79,47,29,16,52,98,86,23  26,61,44,34,92,51,55,24,51,25,23,20,89,29,19,39,97,51,5,43
chr1    3207317 3213609 -       Xkr4    LO_DATA 1,0,2,1,1,2,1,2,4,3,2,0,4,3,5,2,1,0,1,0,3,2,3,1 1,0,0,3,1,2,1,0,1,1,3,0,3,1,0,1,0,0,1,3
```
It lists all introns individually along with their genomic location, status of the test (*OK* or *LO_DATA*), the raw read counts for the intron in all samples, and the model fitted read counts, grouped by condition. If the '--raw-counts-only' option is used, only _raw_ read counts are reported. This information can be used, for instance, to further filter introns with low support across all or subsets of the samples, or from genes with low expression levels, or can be used to generate [visualizations](#visualization) such as heatmaps or PCA plots.

The file *group_data.txt* contains supporting data for all introns in groups tested by MntJULiP.
```
examples
```
It lists all introns in a group, on separate lines, along with their genomic location, status of the test (*OK* or *LO_DATA*), and per sample PSI values, both calculated from the raw read counts and estimated by the model, separated by condition. If the '--raw-counts-only' option is used, only PSI values calculated from the _raw_ read counts are reported. As with intron data, this information can be used to filter low ratio isoforms or to generate [visualizations](#visualization).

### <a name="visualization"></a> Visualization

The output generated by MntJULiP DSR and DSA can be further visualized using the [Jutils](https://github.com/splicebox/Jutils/) visualization suite. Jutils can generate heatmaps of differentially spliced events, sashimi plots, PCA plots and Venn diagrams of gene sets determined to be differentially spliced based on program or user-defined criteria, using information extracted directly from the MntJULiP output.

### <a name="support"></a> Support
Contact: Please submit an [Issue](https://github.com/splicebox/MntJULiP/issues) through this Github page.

#### License information
See the file LICENSE for information on the history of this software, terms
& conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.
