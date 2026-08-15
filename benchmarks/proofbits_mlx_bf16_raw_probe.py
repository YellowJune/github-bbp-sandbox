import json
import numpy as np
import mlx.core as mx
from mlx_lm import load

MODEL='mlx-community/gemma-3-270m-bf16'
model,_=load(MODEL)
mx.eval(model.parameters())
w=model.lm_head.weight if hasattr(model,'lm_head') else model.model.embed_tokens.weight
mx.eval(w)
info={'dtype':str(w.dtype),'shape':list(w.shape),'nbytes':int(w.nbytes)}
try:
    mv=memoryview(w)
    info['memoryview_format']=mv.format
    info['memoryview_nbytes']=mv.nbytes
    raw=np.frombuffer(mv.cast('B'),dtype=np.uint16)
    info['raw_ok']=True
    info['raw_size']=int(raw.size)
    info['raw_first']=[int(x) for x in raw[:16]]
    # Numerical cross-check: BF16 raw bits are the high 16 bits of FP32.
    f32_bits=(raw[:16].astype(np.uint32)<<16)
    decoded=f32_bits.view(np.float32)
    vals=np.array(w.reshape((-1,))[:16].astype(mx.float32),copy=True)
    info['decoded']=[float(x) for x in decoded]
    info['mlx_values']=[float(x) for x in vals]
    info['bit_decode_exact']=bool(np.array_equal(decoded.view(np.uint32),vals.view(np.uint32)))
except Exception as e:
    info['raw_ok']=False
    info['error']=repr(e)
print(json.dumps(info,indent=2))
