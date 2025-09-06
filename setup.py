from setuptools import setup, find_packages

setup(
    name='DePass',
    version='0.0.24',
    description='DePass: a dual-enhanced graph learning framework for paired data integration',
    packages=find_packages(),
    install_requires=[
        'anndata>=0.9.2',        
        'hnswlib>=0.8.0',
        'matplotlib>=3.7.1,<4.0.0',  
        'numpy>=1.23.5,<2.0.0',      
        'pandas>=1.5.3,<2.0.0',
        'scanpy>=1.9.8,<2.0.0',
        'scikit-learn>=1.3.0,<2.0.0',
        'scipy>=1.10.1,<2.0.0',
        'seaborn>=0.12.2,<0.14.0',
        'tqdm>=4.65.0,<5.0.0',
        'jupyter',                   
    ],
    python_requires='>=3.8,<3.9',   
)