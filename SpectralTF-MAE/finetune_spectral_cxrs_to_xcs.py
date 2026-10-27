"""Downstream fine-tuning entry point for complete CXRS -> XCS reconstruction.

The implementation is shared with ``finetune_spectral_xcs_to_cxrs``; this
wrapper reverses the input/output domains while sharing the optimization,
fixed-peak loss/metrics, visualization, and checkpoint implementation.
"""

import finetune_spectral_xcs_to_cxrs as _base

_base_get_args = _base.get_args


def get_args():
    args = _base_get_args()
    args.in_domains = "cxrs"
    args.out_domains = "xcs"
    args.decoder_context_tasks = "cxrs-xcs"
    if args.best_metric == "cxrs_peak_r2":
        args.best_metric = "xcs_peak_r2"
    return args


def main():
    _base.get_args = get_args
    return _base.main()


if __name__ == "__main__":
    main()
