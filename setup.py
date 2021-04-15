import setuptools
import glob

scripts = glob.glob('bin/*.py') + glob.glob('bin/*.sh')
setuptools.setup(scripts=scripts)
