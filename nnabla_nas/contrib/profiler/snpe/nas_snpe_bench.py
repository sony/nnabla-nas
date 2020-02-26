import argparse
import os
import sys
import re
import struct
import glob
import tempfile
import shutil
from tqdm import tqdm
import collections
import numpy as np
from subprocess import Popen, PIPE, STDOUT, check_output
from collections import OrderedDict
import onnx
import json
import time
import logging

import nnabla as nn
from nnabla.utils.nnp_graph import NnpLoader

from nas_helpers import run_command


config = {
    "Name": "",
    "HostRootPath": "",
    "HostResultsDir": "",
    "DevicePath": "/data/local/tmp/snpebenchmark",
    "Devices": ["QV714AE41T"],
    "HostName": "localhost",
    "Runs": 5,
    "Model": {
        "Name": "",
        "Dlc": "",
        "Data": "",
        "InputList": ""
    },
    "Runtimes": ["CPU", "GPU", "GPU_FP16", "DSP"],
    "Measurements": ["timing"],
    "PerfProfile": "default",
    "ProfilingLevel": "detailed",
    "BufferTypes": ["float"]
}


class NnpSnpeBench(object):

    def __init__(self, nnp_file, tempdir, args):
        self.nnp_file = nnp_file
        self.tempdir = tempdir
        self.args = args

        self.nnp_name = self.get_nnp_name(self.nnp_file)
        self.onnx_file = "{}.onnx".format(self.nnp_name)
        self.dlc_file = "{}.dlc".format(self.nnp_name)

    def get_nnp_name(self, nnp_file):
        nnp_name = '{}'.format(os.path.splitext(os.path.basename(nnp_file))[0])
        return nnp_name

    def nnp_to_onnx(self):
        cmdline = "bash -i -c 'conda activate nnabla-build && " \
            "nnabla_cli convert -d opset_6x -b 1 {} {} && " \
            "exit'"\
            .format(self.nnp_file, os.path.join(self.tempdir, self.onnx_file))
        run_command(cmdline)

    def onnx_to_dlc(self):
        ## cmdline = 'bash -i -c "conda activate snpe-env && ' \
        ##   'snpe-onnx-to-dlc --input_network {} --output_path {} --disable_batchnorm_folding && ' \
        # 'exit"'\
        # .format(os.path.join(self.tempdir, self.onnx_file), os.path.join(self.tempdir, self.dlc_file))
        cmdline = 'bash -i -c "conda activate snpe-env && ' \
            'snpe-onnx-to-dlc --input_network {} --output_path {} && ' \
            'exit"'\
            .format(os.path.join(self.tempdir, self.onnx_file), os.path.join(self.tempdir, self.dlc_file))
        run_command(cmdline)

    def create_random_inputs(self):
        # From snpebm_config.py
        dlc_cmd = 'snpe-dlc-info -i "{}"'.format(os.path.join(self.tempdir, self.dlc_file))
        try:
            dlc_info_output = check_output(dlc_cmd, shell=True).decode("utf-8")
        except Exception as de:
            print("Failed to parse {0}".format(dlc_cmd))
            print(de)
        inputs = []  # like ['x0:1,16,16,36', 'x1:1,16,16,36']
        for line in dlc_info_output.split('\n'):
            if ("------------------" in line) or ("Id" in line) or ("Training" in line) or ("Concepts" in line):
                continue
            split_line = line.replace(" ", "").split("|")
            if len(split_line) > 6 and split_line[3] == "data":
                layer_name = split_line[2]
                input_dimensions = split_line[6].replace("x", ",")
                inputs.append("{}:{}".format(layer_name, input_dimensions))
        # Create dir
        input_dir = os.path.join(self.tempdir, "random_inputs")
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        os.makedirs(input_dir)
        # Generate random_raw
        for inp in inputs:
            sinp = inp.split(":")
            name = sinp[0]
            shape = tuple([int(i) for i in sinp[1].split(",")])
            for i in range(args.model_random_input):
                fname = "random_{}_{}.raw".format(name, i)
                with open(os.path.join(input_dir, fname), "wb") as fp:
                    dtype = eval("np.{}".format(args.input_data_type))
                    data = np.random.uniform(-1.0, +1.0, shape).astype(dtype)
                    fp.write(data)
        # Generate random list
        random_list_path = os.path.join(input_dir, "random_raw_list.txt")
        with open(random_list_path, "w") as fp:
            for i in range(args.model_random_input):
                line = []
                for inp in inputs:
                    sinp = inp.split(":")
                    name = sinp[0]
                    fname = "random_{}_{}.raw".format(name, i)
                    dpath = "{}:={}".format(name, os.path.join("random_inputs", fname))
                    line.append(dpath)
                line = " ".join(line)
                fp.write(line)
                fp.write("\n")
        return [input_dir], random_list_path

    def bench_mark(self):
        random_input_dir, random_list_path = self.create_random_inputs()

        config["Name"] = args.name
        config["HostResultsDir"] = "{}/{}".format(args.name, self.nnp_name)
        config["Devices"] = args.devices
        config["Model"]["Name"] = args.model_name
        config["Model"]["Dlc"] = os.path.join(self.tempdir, self.dlc_file)
        config["Model"]["Data"] = random_input_dir
        config["Model"]["InputList"] = random_list_path
        config["Measurements"] = args.measurements
        config["PerfProfile"] = args.perf_profile
        config["ProfilingLevel"] = args.profiling_level
        config_file = "{}_config.json".format(self.nnp_name)
        config_path = os.path.join(self.tempdir, config_file)
        with open(config_path, "w") as fp:
            json.dump(config, fp)

        cmdline = 'bash -i -c "conda activate snpe-env && ' \
            'snpe-bench.py -json -c "{}" && '\
            'exit"'\
            .format(config_path)

        msg = \
            "Takes several minutes.\n" \
            "See the intermediate benchmark commands by watch -n 2 'cat snpe-bench_cmds.sh'\n"
        print(msg)
        run_command(cmdline)

    def bench_nnp(self):
        self.nnp_to_onnx()
        self.onnx_to_dlc()
        self.bench_mark()


def main(args):
    nn.logger.setLevel(logging.ERROR)

    nnp_files = sorted(glob.glob("{}/*.nnp".format(args.nnp_dir)))
    nnp_file = ""
    try:
        for nnp_file_ in nnp_files:
            nnp_file = nnp_file_
            tempdir = tempfile.mkdtemp()

            bench = NnpSnpeBench(nnp_file, tempdir, args)
            bench.bench_nnp()

            shutil.rmtree(tempdir)
    except Exception as e:
        print(e)
        raise RuntimeError("snpe-bench ({}) failed.".format(nnp_file))
    finally:
        if os.path.exists(tempdir):
            shutil.rmtree(tempdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SNPE Latency Measurements for NAS.")
    parser.add_argument('--nnp-dir', type=str,
                        help='Directory for NNP input files.')
    parser.add_argument('--devices', nargs='+',
                        help='Paramter for config.json. Specify device-id!')
    parser.add_argument('--name', type=str, required=True,
                        help='Name is used as prefix of the results. Result directory is like `result_<name>`')
    parser.add_argument('--model-name', type=str, default="SampleNet",
                        help='Paramter for config.json')
    parser.add_argument('--model-random-input', type=int, default=10,
                        help='Paramter for config.json')
    parser.add_argument('--measurements', type=str, nargs='+', default=["timing"],
                        help='Paramter for config.json')
    parser.add_argument('--perf-profile', type=str, default="default",
                        choices=["balanced", "default", "sustained_high_performance",
                                 "high_performance", "power_saver", "system_settings"],
                        help='Paramter for config.json')
    parser.add_argument('--profiling-level', type=str, default="detailed", choices=["basic", "detailed"],
                        help='Paramter for config.json')
    parser.add_argument('--input-data-type', type=str, default="float32",
                        choices=["float32", "uint8"],
                        help='Data type for the random input.')
    args = parser.parse_args()

    main(args)
