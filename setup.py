import setuptools
import os
import glob


def readme():
    with open('README.md') as f:
        return f.read()


def find_scripts():
    scripts = glob.glob(os.path.join('bin', '*.py'))
    scripts.extend(glob.glob(os.path.join('bin', '*.sh')))
    return scripts


setuptools.setup(
    name='cmr_cfr',
    version='0.1.0',
    description='Package for analysis of categorized free recall data.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    author='Neal Morton',
    author_email='mortonne@gmail.com',
    license='GPLv3',
    url='http://github.com/mortonne/cmr_cfr',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    scripts=find_scripts(),
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
    ]
)
