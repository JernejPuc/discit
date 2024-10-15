import os
from setuptools import setup


with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as f:
    readme = f.read()

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='discit',
    version='0.2.0',
    description='Discit: Deep Learning Tools',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/jernejpuc/discit',
    author='Jernej Puc',
    author_email='jernej.puc@fs.uni-lj.si',
    license='MPL 2.0',
    classifiers=[
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'],
    platforms=['Linux'],
    packages=['discit'],
    package_dir={'': 'src'},
    python_requires='~=3.8',
    install_requires=requirements,
    zip_safe=False)
