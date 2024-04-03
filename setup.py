from setuptools import setup

VERSION = '0.0.1'
DESCRIPTION = 'AI based optimized parameter estimation for ML models.'
LONG_DESCRIPTION = 'AI based optimized parameter estimation of ML models using Hybrid of Genetic Algorithm and Simulated Annealing. based on the paper by Jegadit S Saravanan @ https://ieeexplore.ieee.org/document/10308077'


setup(
    name='ml_optimizer',
    version=VERSION,
    packages=['jss_optimizer'],
    install_requires=['numpy', 'scikit-learn', 'tqdm'],
    author='Jegadit S Saravanan',
    author_email='jegaditssaravanan@yahoo.com',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    keywords='machine-learning optimization genetic-algorithm simulated-annealing',
    # url='https://github.com/your_username/ml_optimizer',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)