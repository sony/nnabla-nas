# search DARTS
python main.py  search \
                --device-id 0 --context 'cudnn' \
                --minibatch-size 16 \
                --num-cells 8 \
                --num-nodes 4 \
                --init-channels 16 \
                --mode full \
                --shared-params \
                --config-file examples/darts_search.json

# validate DARTS
python main.py  train \
                --device-id 2 --context 'cudnn' \
                --minibatch-size 48 \
                --num-cells 20 \
                --num-nodes 4 \
                --init-channels 36 \
                --mode full \
                --shared-params \
                --auxiliary \
                --config-file examples/darts_validate.json

# search PNAS
python main.py  search \
                --device-id 1 --context 'cudnn' \
                --minibatch-size 32 \
                --num-cells 8 \
                --num-nodes 4 \
                --init-channels 16 \
                --mode sample \
                --config-file examples/pnas_search.json

# validate PNAS
python main.py  train \
                --device-id 1 --context 'cudnn' \
                --minibatch-size 32 \
                --num-cells 8 \
                --num-nodes 4 \
                --init-channels 48 \
                --mode sample \
                --config-file examples/pnas_valiate.json
