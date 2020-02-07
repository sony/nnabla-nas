# search DARTS
python main.py -d 2 -f examples/darts_search.json --search -a DartsSearcher -o log/darts/search

# validate DARTS
python main.py -d 1 -f examples/darts_validate.json

# search PNAS
python main.py -d 1 -f examples/pnas_search.json

# validate PNAS
python main.py -d 2 -f examples/pnas_validate.json

# search PNAS with constraints
python main.py -d 2 -f examples/constraint_pnas_search.json

# validate PNAS with constraints
python main.py -d 2 -f examples/constraint_pnas_validate.json