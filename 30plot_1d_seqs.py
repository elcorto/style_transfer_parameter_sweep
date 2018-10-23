#!/usr/bin/python3

from matplotlib import pyplot as plt
import numpy as np

from psweep import psweep as ps

import common


def nearest_idx(src, tgt):
    return [np.abs(src - x).argmin() for x in tgt]


if __name__ == '__main__':

    df = ps.df_read('results.pk')
    df = df[df.fail_state.isna()]

    # all possible parameters, with data limits
    vary_cols = dict(
        style_weight = [1, 30],
        tv_weight = None,
        learning_rate = None,
        style_scales = [None, 2],
        content_weight_blend = [None, 0.5],
        style_layer_weight_exp = None,
        )

    img_dct = {key:common.jpegstr2imgarr(val) for key,val
               in common.pkread('img_dct_rgb.pk').items()}
    shape = list(img_dct.values())[0].shape

    n_img_cols = 5
    n_img_rows = len(vary_cols)
    scale = 1/400
    # adjust figsize to close gaps and account for ylabel, which changes all
    # whitespaces again, need to adjust row_fac and left at the same time --
    # yuck! W/o ylabel, we have left=0 and row_fac = 1
    left = 0.25
    row_fac = 0.86
    figsize = (scale * n_img_rows * shape[0] * row_fac,
               scale * n_img_cols * shape[1])
    fig, axs = plt.subplots(n_img_rows, n_img_cols,
                            figsize=figsize,
                            squeeze=False, tight_layout=False,
                            gridspec_kw=dict(hspace=0,
                                             wspace=0,
                                             right=1,
                                             bottom=0,
                                             top=1,
                                             left=left,
                                             width_ratios=[1]*n_img_cols,
                                             height_ratios=[1]*n_img_rows))
    for ii, item in enumerate(vary_cols.items()):
        study, limit = item
        this_df = df[df.study==study].sort_values(study)
        if limit:
            if limit[0]:
                this_df = this_df[this_df[study] >= limit[0]]
            if limit[1]:
                this_df = this_df[this_df[study] <= limit[1]]
        vals = this_df[study].values
##        idxs = np.unique(nearest_idx(vals,
##                                     np.linspace(vals.min(),
##                                                 vals.max(),
##                                                 n_img_cols)))
        idxs = np.unique(np.linspace(0, len(this_df)-1, n_img_cols, dtype=int))
        pset_ids = this_df._pset_id[idxs].values
        for jj,pset_id in enumerate(pset_ids):
            ax = axs[ii,jj]
            ax.imshow(img_dct[pset_id])
            if jj == 0:
                ax.set_ylabel(study,
                              rotation='horizontal',
                              horizontalalignment='right',
                              verticalalignment='center')
            # ax.set_axis_off() ain't good enough, we need to keep the ylabel
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.savefig('scan.jpg', dpi=200)
    plt.show()
