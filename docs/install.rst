DePass Installation
===================

**It is preferred to create a new environment for DePass.**

.. code-block:: bash

   # Create and activate a new conda environment
   conda create -n DePass python==3.8.20
   conda activate DePass

----

**Install DePass from GitHub**

.. code-block:: bash

   git clone https://github.com/zhanglabNKU/DePass.git
   cd DePass
   pip install DePass-0.0.25-py3-none-any.whl

**Additional Dependencies**


Because DePass leverages ``mclust`` for clustering, installing R, the ``rpy2`` Python interface, and the ``mclust`` R package is recommended.

.. code-block:: bash

   conda install -c conda-forge r-base rpy2
   conda install conda-forge::r-mclust

----

**Install PyTorch and PyTorch Geometric**


.. code-block:: bash

   pip install torch==2.4.1
   pip install torch-geometric==2.3.1
   pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
