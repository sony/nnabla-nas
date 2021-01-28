.. _installation:

Installation
============

Run the following code snippet for a local installation:

::
    
    git clone git@github.com:sony/nnabla-nas.git
    cd nnabla_nas
    pip install -r requirements.txt


To build documentation in various formats, you will need `Sphinx <http://www.sphinx-doc.org>`_ and the readthedocs theme.


::

    cd docs/
    pip install -r requirements.txt

You can then build the documentation by running ``make <format>`` from the ``docs/`` folder. Run ``make`` to get a list of all available output formats.