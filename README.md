## MntJULiP ##

MntJULiP is a program for comprehensive and accurate quantification of splicing differences from RNA-seq data. MntJULiP can detect changes in splicing ratios and in absolute splicing levels. It characterizes splicing at the intron level, avoiding the need for pre-assembly, and can be used with or without a reference annotation. This covariate branch contains the beta version, which incorporates the treatment of covariates for large and complex data. 

For the original version of MntJULiP, please refer to the [master](https://github.com/splicebox/MntJULiP/tree/master) branch.

Copyright (C) 2019-2023, and GNU GPL v3.0, by Wui Wang Lui, Guangyu Yang, Liliana Florea

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.  

### <a name="table-of-contents"></a> Table of contents
- [What is MntJULiP?](#what-is-mntjulip)
- [Installation](#installation)
- [Usage](#usage)
- [Input/Output](#inputoutput)
- [Support](#support)

### <a name="what-is-mntjulip"></a> What is MntJULiP?
MntJULiP is a high-performance Python package designed for comprehensive and accurate quantification of splicing differences from RNA-seq data. MntJULiP is assembly-free and can perform intron-level differential splicing analysis from large collections of RNA-seq samples.

MntJULiP works on aligned RNA sequencing reads (generated by Tophat or STAR), using Bayesian mixture models to detect changes in splicing ratios and in absolute splicing levels. Its statistical underpinnings include a Dirichlet multinomial mixture model, to test for differences in the splicing ratio, and a zero-inflated negative binomial mixture model, to test for differential splicing abundance.

#### Features  
- A novel assembly-free model for detecting differential intron abundance and differential intron splicing ratio across samples;
- Allows multi-way comparisons;  
- Incorporates the treatment of covariates within the models;
- Can be used with or without a reference annotation;  
- Discovers more splicing variation than other programs;
- Multi-threaded and highly scalable, can process hundreds of samples in hours.

### <a name="installation"></a> Installation
MntJULiP is written in Python, you can install the latest version from our GitHub repository. To download the code, you can clone this repository by

```
git clone https://github.com/splicebox/MntJULiP.git
```

#### System requirement
* Linux   
* Python 3.7 or later

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
  -v, --version         show program's version number and exit
  -h, --help            show this help message and exit
```

Here is an example to run MntJULiP with a set of alignment files and the GENCODE annotation.
```
ANNO_FILE="gencode.v22.annotation.gtf"
BAM_LIST="bam_file_list.txt"
python3 run.py --bam-list ${BAM_LIST} \
               --anno-file ${ANNO_FILE} \
               --threads 8            
```

The 'bam_list' is a .txt file with columns separated by tabs ('\t'); covariate columns are optional. Here is an example,
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

### <a name="inputoutput"></a> Input/Output
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

For more Input/Output information, please refer to the [master](https://github.com/splicebox/MntJULiP/tree/master#inputoutput) branch.

### <a name="support"></a> Support
Contact: wlui3@jhu.edu, florea@jhu.edu  

#### License information
See the file LICENSE for information on the history of this software, terms
& conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.
