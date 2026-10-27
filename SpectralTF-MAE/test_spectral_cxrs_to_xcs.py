"""Downstream testing entry point for complete CXRS -> XCS reconstruction."""

import test_spectral_xcs_to_cxrs as _base

_base_get_args = _base.get_args


def get_args():
    args = _base_get_args()
    args.in_domains = "cxrs"
    args.out_domains = "xcs"
    args.decoder_context_tasks = "cxrs-xcs"
    return args


def main():
    _base.get_args = get_args
    return _base.main()


if __name__ == "__main__":
    main()
