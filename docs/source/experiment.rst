Experiments
===========

To run an experiment with NNablaNAS, one should create a configuration file (in ``json`` format). Below is an example of how this configuration file look like.

.. image:: images/experiment_cfg.png
    :width: 500
    :align: center


::

    python main.py -d 1 --search \
               -f examples/darts_search.json  \
               -a DartsSearcher \
               -o log/darts/search

:: 

    python main.py -d 1 \
               -f examples/darts_train.json \
               -a Trainer -o log/darts/train


:: 

    mpirun -n 4 main.py -d 0,1,2,3 --search \
               -f examples/darts_search.json  \
               -a DartsSearcher \
               -o log/darts/search
