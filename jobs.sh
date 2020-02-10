# search DARTS
python main.py -d 1 --search \
               -f examples/darts_search.json  \
               -a DartsSearcher \
               -o log/darts/search

# train DARTS
python main.py -d 1 \
               -f examples/darts_train.json \
               -a Trainer -o log/darts/train

# search PNAS
python main.py -d 2 --search \
               -f examples/pnas_search.json \
               -a ProxylessNasSearcher \
               -o log/pnas/search

# train PNAS
python main.py -d 2 \
               -f examples/pnas_train.json \
               -a Trainer \
               -o log/pnas/train

# search PNAS with latency constraints
python main.py -d 3 --search \
               -f examples/pnas_search_latency.json \
               -a ProxylessNasSearcher \
               -o log/pnas-lat/search

# train PNAS with latency constraints
python main.py -d 3 \
               -f examples/pnas_train_latency.json \
               -a Trainer \
               -o log/pnas-lat/train


# search DARTS
python main.py -d 1 --search \
               -f examples/darts_search.json  \
               -a DartsSearcher \
               -o log/darts/search

# search mnv2
python main.py -d 1 --search \
               -f examples/mnv2_search.json  \
               -a DartsSearcher \
               -o log/mnv2/search

python main.py -d 1 \
               -f examples/mnv2_train.json \
               -a Trainer -o log/mnv2/train