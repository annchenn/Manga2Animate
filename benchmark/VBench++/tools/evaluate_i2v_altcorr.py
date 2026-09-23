#!/usr/bin/env python3
"""Experimental VBench-I2V runner using RAFT's bundled alternate correlation.

The standard evaluate_i2v.py and metric formula remain unchanged.  This wrapper
only switches the RAFT implementation flag that upstream RAFT provides for a
slower, substantially lower-memory correlation forward pass.
"""

from easydict import EasyDict

import vbench.dynamic_degree as dynamic_degree


def alternate_corr_args(values):
    values = dict(values)
    values["alternate_corr"] = True
    return EasyDict(values)


dynamic_degree.edict = alternate_corr_args

import evaluate_i2v


if __name__ == "__main__":
    evaluate_i2v.main()
