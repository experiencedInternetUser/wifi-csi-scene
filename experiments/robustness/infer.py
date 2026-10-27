"""Inference with saved classical or TCN predictors on amplitude [N,52,400]."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .modeling import predict


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("amplitudes",type=Path)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    x=np.load(args.amplitudes,allow_pickle=False)
    assert x.ndim==3 and x.shape[1:]==(52,400) and np.isfinite(x).all() and (x>=0).all()
    with threadpool_limits(limits=1):p=predict(joblib.load(args.model),x)
    np.savez_compressed(args.output,probability=p,prediction=(p>=.5).astype(np.int8))
