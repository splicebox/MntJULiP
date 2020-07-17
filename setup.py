"""Setup script for DeepVarint."""
import os
from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import pip


class MyInstall(install):
    def run(self):
        if self.user:
            pip.main(['install', '--user'] + dependencies)
        else:
            pip.main(['install'] + dependencies)
        subprocess.check_output(['cmake', '--version'])
        subprocess.check_call(['make'], cwd='./src')
        subprocess.check_call(['make', 'clean'], cwd='./src')
        subprocess.check_output(['python3', 'build_modules.py'])
        install.run(self)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# required pkgs
dependencies = ['numpy', 'scipy', 'pandas', 'pystan', 'statsmodels', 'dask[complete]']

setup(
    name='mntjulip',
    version='1.1',
    author="Guangyu Yang & Liliana Florea",
    author_email="gyang22@jhu.edu",
    description = ("MntJULiP is a program for comprehensive and accurate quantification of splicing differences from RNA-seq data"),
    license="GNU GPL v3.0",
    # keywords = "",
    url="github.com/splicebox/MntJULiP",
    long_description=read('README.md'),
    include_package_data=True,
    packages=find_packages(),
    # install_requires=dependencies,
    cmdclass={
        'install': MyInstall,
    }
)
