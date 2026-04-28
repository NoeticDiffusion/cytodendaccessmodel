Installation
============

Python ``>=3.10`` is required.

For reviewer reproduction:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install -e ".[dev,viz]"

For local documentation builds:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python -m sphinx -b html docs docs/_build/html

For article or PDF helper tooling that is not part of the Sphinx build:

.. code-block:: bash

   python -m pip install -e ".[article]"

CPU execution is sufficient for the documented reviewer workflow. CUDA is not
required.
