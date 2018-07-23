from setuptools import setup
import versioneer

setup(name='PDPbox',
      packages=['pdpbox'],
      package_data={'pdpbox': ['datasets/*.pkl']},
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='python partial dependence plot toolbox',
      author='SauceCat',
      author_email='jiangchun.lee@gmail.com',
      url='https://github.com/SauceCat/PDPbox',
      license='MIT',
      classifiers = [],
      install_requires=[
          'pandas',
          'numpy',
          'scipy',
          'matplotlib>=2.1.2',
          'joblib',
          'psutil',
          'scikit-learn'
      ],
      zip_safe=False)
