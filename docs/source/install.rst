Installation
============

Transformer-PhysX is currently a pure Python 3 module, meaning that installation should be universal across all platforms.
Depending on your platform, manual installation of `PyTorch <https://pytorch.org/>`_ may be beneficial to ensure cuda version is correct.

Install from PyPI (recommended)
-------------------------------
.. code-block:: python

 pip install trphysx

PyPI `homepage <https://pypi.org/project/trphysx/>`_ and previous `versions <https://pypi.org/project/trphysx/#history>`_.


Install from Source
-------------------
.. code-block:: bash

 git clone https://github.com/zabaras/transformer-physx.git

 cd transformer-physx/

 pip install -e .

If you want to change the module's source code or want the latest pushed commit.

Dependencies
-------------------
For the most up-to-date list of dependencies, please see the `setup.py <https://github.com/zabaras/transformer-physx/blob/main/setup.py>`_. The general list is:

- `torch >= 1.7.0 <https://pypi.org/project/torch/>`_
- `filelock >= 3.0.0 <https://pypi.org/project/filelock/>`_
- `h5py >= 2.9.0 <https://pypi.org/project/h5py/>`_
- `numpy >= 1.15.0 <https://pypi.org/project/numpy/>`_ (should already be installed with torch)
- `matplotlib >= 3.0.0 <https://pypi.org/project/matplotlib/>`_ (for visualizations)



Verify Installation
-------------------
.. code-block:: python
 
 python -c 'import trphysx; print(trphysx.__version__)' 