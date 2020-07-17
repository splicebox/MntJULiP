"""Setup script for DeepVarint."""
import os
from pathlib import Path
from setuptools import setup, find_packages
from distutils.command.clean import clean
from distutils.command.install import install as DistutilsInstall
import subprocess

from models import init_null_BN_model, init_alt_BN_model, init_null_DM_model, init_alt_DM_model


class MyInstall(DistutilsInstall):
    def run(self):
        subprocess.check_output(['cmake', '--version'])
        subprocess.check_call(['make'], cwd='./src')
        subprocess.check_call(['make', 'clean'], cwd='./src')
        base_dir = os.path.dirname(os.path.abspath(__file__))

        model_dir = Path(base_dir) / 'lib'
        model_dir.mkdir(parents=True, exist_ok=True)
        init_null_BN_model(model_dir)
        init_alt_BN_model(model_dir)
        init_null_DM_model(model_dir)
        init_alt_DM_model(model_dir)

        DistutilsInstall.run(self)


class CleanCommand(clean):
    """Custom clean command to tidy up the project root."""
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# required pkgs
dependencies = ['numpy', 'scipy', 'pandas', 'pystan', 'statsmodels', 'dask[complete]', 'pysam']

setup(
    name='MntJULiP',
    version='1.0',
    author="Guangyu Yang & Liliana Florea",
    author_email="gyang22@jhu.edu",
    description = ("MntJULiP is a program for comprehensive and accurate quantification of splicing differences from RNA-seq data"),
    license="GNU GPL v3.0",
    # keywords = "",
    url="github.com/splicebox/MntJULiP",
    # packages=['mntjulip'],
    long_description=read('README.md'),
    include_package_data=True,
    packages=find_packages(),
    install_requires=dependencies,
    cmdclass={
        'clean': CleanCommand,
        'install': MyInstall
    }
)
