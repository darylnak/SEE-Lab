import os
import re
import tables
import urllib.request
import logging
import pyedflib
import requests
import tempfile
from io import StringIO
import subprocess

import numpy as np
import pandas as pd

LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
ch.setLevel(logging.INFO)
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

URL_STUB = "https://alpha.physionet.org/static/published-projects/chbmit/1.0.0/"

OUT_DIR_STUB = os.environ.get("OUT_DIR_STUB", "../input/physionet/")

FILTERS = tables.Filters(complib="blosc", complevel=9)
OUTPATH = OUT_DIR_STUB + "/eeg_data_temples2.h5"
# not all files have the same channels
CHANNELS = [
"FP1-F7", 
"F7-T7", 
"T7-P7", 
"P7-O1", 
"FP1-F3", 
"F3-C3", 
"C3-P3", 
"P3-O1", 
"FP2-F4", 
"F4-C4", 
"C4-P4", 
"P4-O2", 
"FP2-F8", 
"F8-T8", 
"T8-P8", 
"P8-O2", 
"FZ-CZ", 
"CZ-PZ", 
"P7-T7", 
"T7-FT9", 
"FT9-FT10", 
"FT10-T8", 
"T8-P8"]

BUFFER = 300

pd.options.mode.chained_assignment = None


def main():
    subjects = ["chb{0:02d}".format(x) for x in range(1,25)]

    if not os.path.exists(OUT_DIR_STUB):
        os.makedirs(OUT_DIR_STUB)

    h5_file = tables.open_file(OUTPATH, "w")
    for s in subjects:
        process_subject(s, h5_file)


def process_subject(subject, h5_file):
    subject_dir = OUT_DIR_STUB + "/raw/{}".format(subject)
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)

    summary_path = subject_dir + "/{0}-summary.txt".format(subject)
    summary_url = URL_STUB + "/{0}/{0}-summary.txt".format(subject)
    download_file(summary_url, summary_path)
    freq, seizure_times = parse_summary_file(summary_path)
    
    if "/" + subject not in h5_file:
        group = h5_file.create_group("/", subject)
    group = h5_file.get_node("/" + subject)

    counter = 0
    num_files = len(seizure_times)
    data_files = sorted(seizure_times.keys(), 
        key=lambda x: int(re.findall("\d+", x.split("_")[1])[0]))
    seizure_ix = 1
    for name in data_files:
        LOG.info("Processing {}/{} for subject: {}".format(counter, num_files, subject))
        
        if "/{}/{}".format(subject, name.replace(".edf","")) in h5_file:
            LOG.info("File exists. Skipping")
            continue

        url = "{}/{}/{}".format(URL_STUB, subject, name)
        path = subject_dir + "/" + name
        path = download_file(url, path)

        fh = pyedflib.EdfReader(path)
        sigbufs = np.zeros((fh.getNSamples()[0], fh.signals_in_file))
        for ix in range(fh.signals_in_file):
            sigbufs[:,ix] = fh.readSignal(ix)
        data = pd.DataFrame(sigbufs, columns=fh.getSignalLabels())
        flag = False
        for c in CHANNELS:
            if c not in data.columns:
                LOG.warning("Channel {} not present. Skipping file".format(c))
                flag = True
                break
        if flag:
            continue
        
        data = data.loc[:,CHANNELS]
        data = data.loc[:,~data.columns.duplicated()]
        freq = fh.getSampleFrequency(0)
        seizure = np.zeros((data.shape[0],))
        timestamp = np.arange(data.shape[0])*(1/freq)
        ixs_in_file = []
        for x in seizure_times[name]:
            where = np.logical_and(timestamp >= x[0], timestamp <= x[1])
            if where.any():
                seizure[where] = seizure_ix
                ixs_in_file.append(seizure_ix)
                if seizure[-1] == 0:
                    seizure_ix += 1
        data["is_seizure"] = seizure
        LOG.info("Table: {} => {} => {}".format(name, ixs_in_file, np.unique(data["is_seizure"])))
        xx = int(BUFFER*(1/freq))
        X = data.values[xx:-xx,:]
        table = h5_file.create_carray(
            group, name.replace(".edf",""), obj=X, filters=FILTERS)
        table.attrs.seizures = ixs_in_file
        table.attrs.colnames = data.columns.values
        counter += 1


def parse_summary_file(path):
    with open(path) as fh:
        lines = fh.read().split("\n")
    
    freq = 1/float(re.findall(r"\d+", lines[0])[0])
    
    filenames = filter(lambda x: "File Name: " in x, lines)
    filenames = [x.split(": ")[1] for x in filenames]
    
    seizure_times = {}
    for fn in filenames:
        s = np.argmax([fn in x for x in lines])
        seizure_times[fn] = []
        while s < len(lines) and len(lines[s]):
            l = lines[s]
            if "Seizure" in l and "Start Time" in l:
                t = int(re.findall(r"\d+", l)[-1])
                seizure_times[fn].append([t,-1])
            if "Seizure" in l and "End Time" in l:
                t = int(re.findall(r"\d+", l)[-1])
                seizure_times[fn][-1][1] = t
            s += 1
    
    return freq, seizure_times


def download_file(url, outpath, replace=False):
    if os.path.exists(outpath) and not replace:
        LOG.info("Path: {} exists. Skipping".format(outpath))
        return outpath
    

    urlh = urllib.request.urlopen(url)
    with open(outpath, "wb") as fh:
        fh.write(urlh.read())
    urlh.close()
    return outpath


if __name__=="__main__":
    main()
