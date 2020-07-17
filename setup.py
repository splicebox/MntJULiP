"""Setup script for DeepVarint."""
import os
from setuptools import setup, find_packages
from distutils.command.clean import clean
from distutils.command.install import install as DistutilsInstall
import subprocess


class MyInstall(DistutilsInstall):
    def run(self):
        import pip
        pip.main(['install', '--user'] + dependencies)
        subprocess.check_output(['cmake', '--version'])
        subprocess.check_call(['make'], cwd='./src')
        subprocess.check_call(['make', 'clean'], cwd='./src')
        DistutilsInstall.run(self)
        subprocess.check_output(['python3', 'build_modules.py'])


class CleanCommand(clean):
    """Custom clean command to tidy up the project root."""
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# required pkgs
dependencies = ['numpy', 'scipy', 'pandas', 'pystan', 'statsmodels', 'dask[complete]']

setup(
    name='mntjulip',
    version='1.0',
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
        'clean': CleanCommand,
        'install': MyInstall
    }
)
