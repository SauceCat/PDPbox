from setuptools import setup
import versioneer

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=required,
)
