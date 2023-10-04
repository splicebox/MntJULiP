## MntJULiP ##

MntJULiP is a program for comprehensive and accurate quantification of splicing differences from RNA-seq data. MntJULiP can detect changes in splicing ratios and in absolute splicing levels. It characterizes splicing at the intron level, avoiding the need for pre-assembly, and can be used with or without a reference annotation.

Copyright (C) 2019-2020, and GNU GPL v3.0, by Guangyu Yang, Liliana Florea

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.  

### <a name="table-of-contents"></a> Table of contents
- [What is MntJULiP?](#what-is-mntjulip)
- [Installation](#installation)
- [Usage](#usage)
- [Input/Output](#inputoutput)
- [Support](#support)

### <a name="what-is-mntjulip"></a> What is MntJULiP?
MntJULiP is a high-performance Python package designed for comprehensive and accurate quantification of splicing differences from RNA-seq data. MntJULiP is assembly-free and can perform intron-level differential splicing analysis from large collections of RNA-seq samples.

MntJULiP works on aligned RNA sequencing reads (generated by Tophat or STAR), using Bayesian mixture models to detect changes in splicing ratios and in absolute splicing levels. Its statistical underpinnings include a Dirichlet multinomial mixture model and a zero-inflated negative binomial mixture model.

#### Features  
- A novel assembly-free model for detecting differential intron abundance and differential intron splicing ratio across samples;
- Allow multi-way comparisons;  
- Can be used with or without a reference annotation;  
- Discover more splicing variation than other programs;
- Multi-threaded and highly scalable, and could process hundreds of samples in hours.

### <a name="installation"></a> Installation
MntJULiP is written in Python, you can install the latest version from our GitHub repository. To download the codes, you can clone this repository by

```
git clone https://github.com/splicebox/MntJULiP.git
```

#### System requirement
* Linux   
* Python 3.7 or later

#### Prerequisites:
MntJULiP has the following dependencies:
* [PyStan](https://pystan.readthedocs.io/), a package for for statistical modeling and high-performance statistical computation.  
* [NumPy](https://numpy.org/), a fundamental package for scientific computing with Python.    
* [SciPy](https://www.scipy.org/), a Python-based package for mathematics, science, and engineering.  
* [Statsmodels](https://www.statsmodels.org/), a Python module for the estimation of different statistical models, conducting statistical tests and data exploration.  
* [Pandas](https://pandas.pydata.org/), a fast, powerful, flexible and easy to use open source data analysis and manipulation tool.  
* [Dask](https://dask.org/), a Python package that provides advanced parallelism for analytics.  
* [scikit-learn](https://scikit-learn.org/), a Python simple and efficient tools for predictive data analysis.

The required packages may be installed using conda:
```
cd MnJULiP2
conda env create -f mntjulip2.yml
#  run setup.py to install MntJULiP and all the required packages
module load cmake
python3 setup.py install  
```

### <a name="usage"></a> Usage
```
Usage: python run.py [options] --bam-list bam_file_list.txt

Options:
  -h, --help            show this help message and exit
  --bam-list BAM_LIST   a text file that contains the list of the BAM file
                        paths and sample conditions.
  --splice-list SPLICE_LIST
                        a text file that contains the list of the SPLICE file
                        paths and sample conditions.
  --anno-file ANNO_FILE
                        annotation file in GTF format.
  --out-dir OUT_DIR     output folder to store the results and temporary
                        files. (default: ./out))
  --num-threads NUM_THREADS
                        number of CPU cores use to run the program. (default: 4)
```

Here is an example to run MntJULiP with Gencode annotation.
```
ref_file="gencode.v22.annotation.gtf"
bam_list="bam_file_list.txt"
python3 run.py --bam-list ${bam_list} \
               --anno-file ${ref_file} \
               --threads 10            
```

The 'bam_list' is an .txt file with two columns separating by 'tab' or '\t'. Here is an example,
```
sample   condition	covariate1
sample1.bam    control	Male
sample2.bam    case	Female
```

Covariate columns are optional.

### <a name="inputoutput"></a> Input/Output
The main input of MntJULiP is a list of BAM files with RNA-Seq read mappings. The BAM files can be generated by STAR with/without '--outSAMstrandField intronMotif' option.
```
STAR --genomeDir ${STARIDX_DIR} \
     --outSAMstrandField intronMotif \
     --readFilesIn ${DATA_DIR}/${name}_1.fastq ${DATA_DIR}/${name}_2.fastq \
     --outSAMtype BAM SortedByCoordinate \
     --outFileNamePrefix ${WORK_DIR}/${name}/
```

As an option, the BAM file can be sorted by genomic location and indexed for random access.  
```
samtools sort -o accepted_hits.sorted.bam accepted_hits.bam
samtools index accepted_hits.sorted.bam
```


### <a name="support"></a> Support
Contact: gyang22@jhu.edu, florea@jhu.edu  

#### License information
See the file LICENSE for information on the history of this software, terms
& conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.
