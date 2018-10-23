#!/usr/bin/python3

import subprocess
import multiprocessing
import os
import re
from itertools import product

import numpy as np
import pandas as pd

from psweep import psweep as ps

pj = os.path.join


# Need to use CUDA_VISIBLE_DEVICES to select CPU device.
#     https://github.com/tensorflow/tensorflow/issues/3644
#     https://github.com/tensorflow/tensorflow/issues/5066
def worker(pset):
    templ = """
    mkdir -p {pset_dir} {pset_dir_checkpoint};
    {gpu_device_str}
    python3 ~/soft/git/neural-style/neural_style.py
        --network ~/soft/git/neural-style/imagenet-vgg-verydeep-19.mat
        --content {content}
        --styles {styles}
        --width {width}
        --output {output}
        --iterations {iterations}
        --style-weight {style_weight}
        --content-weight {content_weight}
        --style-layer-weight-exp {style_layer_weight_exp}
        --print-iterations {print_iterations}
        --learning-rate {learning_rate}
        --style-scales {style_scales}
        --tv-weight {tv_weight}
        --pooling {pooling}
        --progress-plot
        --progress-write
        --preserve-colors
        --content-weight-blend {content_weight_blend}
        {initial_str}
        > {pset_dir}/log 2>&1;
        sleep 5
    """
##        --checkpoint-output '{pset_dir_checkpoint}/out_{{:05}}.png'
##        --checkpoint-iterations {checkpoint_iterations}
    pset['initial_str'] = ''
    if pset['initial'] is not None:
        pset['initial_str'] = '--initial {}'.format(pset['initial'])
    pset['gpu_device_str'] = ''
    if pset['gpu_device'] is not None:
        pset['gpu_device_str'] = 'CUDA_VISIBLE_DEVICES={}'.format(pset['gpu_device'])
    pset_dir = os.path.expanduser(os.path.abspath(pj(pset['_calc_dir'],
                                                     pset['_pset_id'])))
    update = dict(pset_dir=pset_dir,
                  pset_dir_checkpoint=pj(pset_dir, 'checkpoint'),
                  output=pj(pset_dir, 'out.png'),
                  )
    pset.update(update)
    cmd = re.sub(r'\s+', ' ', templ.format(**pset).strip())
    pset['cmd'] = cmd
    subprocess.run(cmd, shell=True, check=True)
    return pset


def gpu_worker(pset):
    name = multiprocessing.current_process().name
    pset['gpu_device'] = int(name.replace('ForkPoolWorker-','')) - 1
    pset.update(worker(pset))
    return pset


def read_old_1d(df, col):
    return df[df.study==col][col].values


if __name__ == '__main__':
    # good defaults for
    #   initial=None
    #   pooling=max
    const = dict(
        print_iterations = 10,
        checkpoint_iterations = 100,
        width = 512,
        iterations = 1000,
        learning_rate = 10,
        style_scales = 1,
        content_weight_blend = 0.2,
        content_weight = 1,
        style_weight = 10,
        tv_weight = 100,
        style_layer_weight_exp = 1,
        content = '~/images/content/selfie.jpg',
        styles = '~/images/style/candy.jpg',
        initial = None,
        pooling = 'max',
        gpu_device = None, # set in gpu_worker()
        )

    vary = dict(
        learning_rate = [0.1, 0.3, 0.6, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50],
        style_scales = np.concatenate((np.linspace(0.1, 1, 19),
                                       np.linspace(1.1, 3, 30))),
        content_weight_blend = np.linspace(0,1,30),
        style_weight = np.linspace(0,200,50),
        tv_weight = np.linspace(0,600,50),
        style_layer_weight_exp = np.concatenate((np.linspace(0,1.3,15),
                                                 np.linspace(1.33,20,20))),
        )

    params = []
    disp_cols = []

    for study,seq_1d in vary.items():
        params_1d = ps.seq2dicts(study, seq_1d)
        this_params = ps.loops2params(product(params_1d, [{'study': study}]))
        this_params = [ps.merge_dicts(const, dct) for dct in this_params]
        params += this_params
        disp_cols.append(study)

    pd.options.display.max_rows = None
    pd.options.display.max_columns = None

    df = ps.run(gpu_worker,
                params,
                poolsize=4,
                simulate=False,
                verbose=disp_cols + ['study'],
                tmpsave=False,
                backup_script=__file__,
                backup_calc_dir=True)
