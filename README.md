## MntJULiP ##

MntJULiP comprehensively and accurately quantifies splicing differences at intron level from collections of RNA-seq data. 

Described in:

Yang G, Sabinciyan S, and Florea L (2021). Comprehensive and scalable quantification of splicing differences with MntJULiP. *Submitted.* 

```
Copyright (C) 2019-2021, and GNU GPL v3.0, by Guangyu Yang, Liliana Florea
```

Data described in the article can be found [here](http://ccb.jhu.edu/software/MntJULiP/). Supplementary [scripts](https://github.com/splicebox/MntJULiP/blob/master/MntJULiP_scripts.tar.gz) are provided with this distribution. 

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.  

### <a name="table-of-contents"></a> Table of contents
- [What is MntJULiP?](#what-is-mntjulip)
- [Installation](#installation)
- [Usage](#usage)
- [Input/Output](#inputoutput)
- [Support](#support)

### <a name="what-is-mntjulip"></a> What is MntJULiP?
MntJULiP is a high-performance Python package for comprehensive and accurate quantification of splicing differences from RNA-seq data. It is designed to detect both changes in splicing ratios (*differential splicing ratio*, DSR), and changes in the absolute splicing levels (*differential splicing abundance*, DSA). Like LeafCutter, MntJULiP works at the level of introns, or splice junctions, and therefore it is assembly-free, and can be used with or without a reference annotation. MntJULiP is fully scalable, and can work with very large RNA-seq data collections, as well as with just a few samples.


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
pip3 install --user numpy scipy pandas pystan statsmodels "dask[complete]"
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
                        annotation file in GTF format.
  --out-dir OUT_DIR     output folder to store the results and temporary files. (default: ./out)
  --num-threads NUM_THREADS
                        number of CPU cores use to run the program. (default: 4)
  -v, --version         show program's version number and exit
  -h, --help            show this help message and exit.
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
The .splice file is a space or ' ' separated file with at least 5 columns "chrom start end count strand" (header is excluded):
```
chr1 1311924 1312018 100 -
```
A splice file may have additional columns; for instance, those generated by the junc tool included with this package will distinguish the numbers of uniquely and multiply mapped supporting reads:
```
chr1 1311924 1312018 100 - 67 33 ...
```

### <a name="inputoutput"></a> Input/Output
The main **input** of MntJULiP is a list of BAM files containing the RNA-seq read alignments sorted by genomic coordinates. Currently, MntJULiP has been adapted to alignments generated with Tophat2 and STAR with/without '--outSAMstrandField intronMotif' option, for example:
```
STAR --genomeDir ${STARIDX_DIR} \
     --outSAMstrandField intronMotif \
     --readFilesIn ${DATA_DIR}/${name}_1.fastq ${DATA_DIR}/${name}_2.fastq \
     --outSAMtype BAM SortedByCoordinate \
     --outFileNamePrefix ${WORK_DIR}/${name}/
```

Alternatively, MntJULiP can take as **input** .splice files generated by an external program, such as the junc tool included with this package, or other versions from our [PsiCLASS](https://github.com/splicebox/PsiCLASS) or [CLASS2](https://sourceforge.net/p/splicebox/wiki/Home/) software.
```
junc my_alignment_file.bam -a > my_alignment_file.splice
```

**Output** generated by MntJULiP includes four types of files:

*intron_data.txt*: contains information about the introns (splice junctions), including genomic coordinates, read counts, average abundance levels, etc;  
*diff_introns.txt*: contains the results of the differential intron abundance (DSA) analysis;  
*diff_spliced_introns.txt* & *diff_spliced_groups.txt*: contains the results of the differential intron splicing ratio (DSR) analysis.  

In addition, the directory will contain a temporary folder named 'data', containing the .splice files generated by MntJULiP.

### <a name="support"></a> Support
Contact: etzlanim@gmail.com, florea@jhu.edu  

#### License information
See the file LICENSE for information on the history of this software, terms
& conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.
