import json, inspect, time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load

MODEL='mlx-community/Qwen2.5-0.5B-Instruct-bf16'
model,tok=load(MODEL)
mx.eval(model.parameters())
attrs=[a for a in dir(model) if not a.startswith('_')]
info={'model':MODEL,'model_type':str(type(model)),'attrs':attrs[:300]}
for name in ['model','lm_head','embed_tokens','layers','norm','args']:
    if hasattr(model,name):
        obj=getattr(model,name);info[f'attr_{name}_type']=str(type(obj));
        try: info[f'attr_{name}_shape']=list(obj.weight.shape) if hasattr(obj,'weight') else None
        except Exception as e: info[f'attr_{name}_shape_error']=repr(e)
# Basic forward.
ids=mx.array([[1,2,3,4]],dtype=mx.int32)
out=model(ids)
mx.eval(out)
info['forward_shape']=list(out.shape);info['forward_dtype']=str(out.dtype)
# Probe inner body if present.
if hasattr(model,'model'):
    try:
        body=model.model(ids);mx.eval(body);info['inner_model_shape']=list(body.shape);info['inner_model_dtype']=str(body.dtype)
    except Exception as e: info['inner_model_error']=repr(e)
# Probe official custom Metal API with a tiny array.
src='''
    uint i = thread_position_in_grid.x;
    if (i < n) out[i] = inp[i] + T(1.0);
'''
try:
    k=mx.fast.metal_kernel(name='pb_probe',input_names=['inp'],output_names=['out'],source=src)
    x=mx.arange(16,dtype=mx.float32)
    y=k(inputs=[x],template=[('T',mx.float32)],grid=(16,1,1),threadgroup=(16,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype],init_value=0.0,ensure_row_contiguous=True)[0]
    mx.eval(y);info['custom_kernel_ok']=bool(mx.all(y==x+1).item())
except Exception as e: info['custom_kernel_error']=repr(e)
# Inspect signatures likely needed for cache/incremental decode.
for target,name in [(model,'model_call'),(getattr(model,'model',None),'inner_call')]:
    if target is not None:
        try: info[name+'_signature']=str(inspect.signature(target.__call__))
        except Exception as e: info[name+'_signature_error']=repr(e)
Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);p=Path('experiments/artifacts/proofbits_mlx_integration_probe.json');p.write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
