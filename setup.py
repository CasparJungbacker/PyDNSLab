from setuptools import setup, find_packages

setup(
    name='PyDNSLab',
    version='0.0.1',
    url='https://github.com/CasparJungbacker/PyDNSLab',
    author='Caspar Jungbacker',
    author_email='caspar.jungbacker@outlook.com',
    description='Python port of DNSLab written for MatLab',
    packages=find_packages(),    
    install_requires=['numpy'],
)