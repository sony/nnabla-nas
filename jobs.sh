# search DARTS
python main.py -d 2 -f examples/darts_search.json --search -a DartsSearcher -o log/darts/search

# search PNAS
python main.py -d 3 -f examples/pnas_search.json --search -a ProxylessNasSearcher -o log/pnas/search
