# search DARTS
python main.py  search \
                --device-id 1 --context 'cudnn' \
                --mini-batch-size 16 \
                --num-cells 8 \
                --num-nodes 4 \
                --init-channels 16 \
                --mode full \
                --shared-params \
                --config-file examples/darts_search.json

# validate DARTS
python main.py  train \
                --device-id 2 --context 'cudnn' \
                --batch-size-train 48 \
                --batch-size-valid 40 \
                --num-cells 20 \
                --num-nodes 4 \
                --init-channels 36 \
                --mode full \
                --shared-params \
                --auxiliary \
                --cutout \
                --config-file examples/darts_validate.json

# search PNAS
python main.py  search \
                --device-id 1 --context 'cudnn' \
                --mini-batch-size 128 \
                --num-cells 8 \
                --num-nodes 4 \
                --init-channels 16 \
                --mode sample \
                --config-file examples/pnas_search.json

# validate PNAS
python main.py  train \
                --device-id 1 --context 'cudnn' \
                --batch-size-train 48 \
                --batch-size-valid 100 \
                --num-cells 8 \
                --num-nodes 4 \
                --init-channels 48 \
                --mode sample \
                --config-file examples/pnas_validate.json
