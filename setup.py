from setuptools import setup

setup(name='PDPbox',
      packages=['pdpbox'],
      version='0.1',
      description='python partial dependence plot toolbox',
      author='SauceCat',
      author_email='jiangchun.lee@gmail.com',
      url='https://github.com/SauceCat/PDPbox',
      download_url = 'https://github.com/SauceCat/PDPbox/archive/0.1.zip',
      license='MIT',
      classifiers = [],
      install_requires=[
          'pandas',
          'numpy',
          'matplotlib>=2.1.2',
          'joblib',
          'psutil',
          'scikit-learn'
      ],
      zip_safe=False)