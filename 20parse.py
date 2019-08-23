#!/usr/bin/env python3

import re
import os

import common

from psweep import psweep as ps

pj = os.path.join


if __name__ == '__main__':

    # jpeg compression quality (percent)
    quality = 40

    # all data, not part of this repo
    basedir = ps.fullpath('~/work/data/hackathon/calc')
    df = ps.df_read(f'{basedir}/results.pk')

    # new column, detect failed run
    df = df.reindex(columns=df.columns.tolist() + ['fail_state'])
    cases = [(r'std::bad_alloc', 'bad_alloc'),
             (r'Killed', 'killed'),
             ]

    img_dct_rgb = {}
    img_dct_gray = {}
    for pset_id in df._pset_id.values:
        with open(pj(basedir, pset_id, 'log')) as fd:
            txt = fd.read()
        go = True
        for regex, fail_state in cases:
            if re.search(regex, txt, re.M) is not None:
                df.loc[df._pset_id==pset_id, 'fail_state'] = fail_state
                go = False
                break
        if go:
            fn = pj(basedir, pset_id, 'out.png')
            img_dct_gray[pset_id] = common.file2jpegstr(fn,
                                                        convert='L',
                                                        quality=quality)
            img_dct_rgb[pset_id] = common.file2jpegstr(fn,
                                                       quality=quality)

    # should be empty!
    print(df[~df.fail_state.isna()])

    # local copy of updated database
    ps.df_write(df, 'results.pk')
    common.pkwrite(img_dct_rgb, 'img_dct_rgb.pk')
    common.pkwrite(img_dct_gray, 'img_dct_gray.pk')
