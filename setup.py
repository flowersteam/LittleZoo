from setuptools import setup, find_packages

setup(
    name='LittleZoo',
    version='0.2.0',
    description='A textual environment to test LLM Agent commonsense knowledge generalization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Loris Gaven',
    url='https://github.com/lorisgaven/LittleZoo',
    packages=find_packages('.'),
    install_requires=[
        'numpy',
        'matplotlib',
        'gymnasium',
        'pygame',
    ]
)
