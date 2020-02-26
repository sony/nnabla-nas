import argparse
import os
import glob
import json
import csv


def create_simple_stats(data):
    runtimes = data["Runtimes"].split(",")
    ed = data["Execution_Data"]
    simple_stats = {}

    for runtime in runtimes:
        fps = ed[runtime]["Forward Propagate"]
        tis = ed[runtime]["Total Inference Time"]
        simple_stats[runtime] = {
            "Forward Propagate": {},
            "Total Inference Time": {},
            "Overhead": {},
            "Layers Time": {},
        }
        for elm0, elm1 in zip(fps.items(), tis.items()):
            k0, v0 = elm0
            k1, v1 = elm1
            key = k0
            if type(v0) == str:
                continue
            simple_stats[runtime]["Forward Propagate"][key] = v0
            simple_stats[runtime]["Total Inference Time"][key] = v1
        edr = ed[runtime]

        latency0, latency1, latency2 = 0.0, 0.0, 0.0
        for key in edr.keys():  # layers loop in modules
            if "layer_" not in key:
                continue
            if "Type:data" in key:
                continue
            latency0 += edr[key]["Min_Time"]
            latency1 += edr[key]["Avg_Time"]
            latency2 += edr[key]["Max_Time"]
        simple_stats[runtime]["Layers Time"]["Min_Time"] = latency0
        simple_stats[runtime]["Layers Time"]["Avg_Time"] = latency1
        simple_stats[runtime]["Layers Time"]["Max_Time"] = latency2
    return simple_stats


def main(args):
    space_latency = args.space_latency
    host_result_json_files = glob.glob(os.path.join(space_latency, "*", "latest_results", "*.json"))

    simple_table = {}
    for i, fpath in enumerate(host_result_json_files):
        with open(fpath) as fp:
            data = json.load(fp)
            uid = fpath.split("/")[1]
            simple_table[uid] = create_simple_stats(data)
    last_data = data

    # JSON
    out_file = "{}.json".format(space_latency)
    with open(out_file, "w") as fp:
        json.dump(simple_table, fp)

    # CSV
    out_file = "{}.csv".format(space_latency)
    with open(out_file, "w") as fp:
        writer = csv.writer(fp)
        for k0 in simple_table:
            for k1 in simple_table[k0]:
                for k2 in simple_table[k0][k1]:
                    for k3 in simple_table[k0][k1][k2]:
                        writer.writerow([k0, k1, k2, k3, simple_table[k0][k1][k2][k3]])

    # Meta info
    out_file = "{}.meta".format(space_latency)
    with open(out_file, "w") as fp:
        writer = csv.writer(fp, delimiter=":")
        writer.writerow(['ro.product.model', last_data['ro.product.model']])
        writer.writerow(['ro.product.board', last_data['ro.product.board']])
        writer.writerow(['ro.product.cpu.abi', last_data['ro.product.cpu.abi']])
        writer.writerow(['ro.build.version.sdk', last_data['ro.build.version.sdk']])
        writer.writerow(['ro.product.manufacturer', last_data['ro.product.manufacturer']])
        writer.writerow(['Date', last_data['Date']])
        writer.writerow(['Architectures', last_data['Architectures']])
        writer.writerow(['Devices', last_data['Devices']])
        writer.writerow(['SNPE SDK version:', last_data['SNPE SDK version:']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NNP Files Generation for NAS.")
    parser.add_argument('--space-latency', type=str,
                        help='Directory path to snpe-bench.py results for unique moduels',
                        required=True)
    args = parser.parse_args()

    main(args)
