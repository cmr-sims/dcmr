import setuptools
import glob

scripts = glob.glob('bin/*.py') + glob.glob('bin/*.sh') + ['bin/dcmr-plan']
setuptools.setup(scripts=scripts)
