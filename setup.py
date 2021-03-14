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
          'joblib',
          'psutil',
          'xgboost==1.3.3',
          'matplotlib==3.1.1',
          'sklearn==0.23.1'
      ],
      zip_safe=False)
