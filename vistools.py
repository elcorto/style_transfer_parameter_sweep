from psweep import psweep as ps
import holoviews as hv
import common


def get_holomaps():
    # dict with jpeg byte strings of all images from the parameter study
    #   {pset_id: jpegstr}
    jpegstr_dct = common.pkread('img_dct_rgb.pk')

    # shape: assume all imgs have the same shape
    shape = common.jpegstr2imgarr(jpegstr_dct[list(jpegstr_dct.keys())[0]]).shape

    # the parameter sweep database (created by the psweep package)
    df = ps.df_read('results.pk')
    df = df[df.fail_state.isna()]

    vary_cols = ['style_weight', 'tv_weight', 'learning_rate', 'style_scales',
                 'content_weight_blend', 'style_layer_weight_exp']

    holos = {}
    print("creating holomaps ...")
    for study in vary_cols:
        print("    " + study)
        this_df = df[df.study==study].sort_values(study)

        # {value of varied param (study): array shape (width, height, 3),...}
        imgs = dict((this_df.loc[this_df._pset_id==pset_id,study][0],
                     hv.RGB(common.jpegstr2imgarr(jpegstr_dct[pset_id])))
                    for pset_id in this_df._pset_id)

        holos[study] = hv.HoloMap(imgs, kdims=study)

    # holoviews settings for matplotlib
    hv.util.opts({'RGB': {'plot': {'fig_latex':False,
                                   'aspect': shape[1]/shape[0],
                                   'fig_size':200,
                                   'xaxis': False,
                                   'yaxis': False}}})

    print("\nhang tight, we're rendering stuff ...")
    return holos
