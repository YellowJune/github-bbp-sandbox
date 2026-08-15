import json, inspect
from pathlib import Path
import mlx.core as mx
from mlx_lm import load

MODEL='mlx-community/Qwen2.5-0.5B-Instruct-bf16'
model,tok=load(MODEL)
mx.eval(model.parameters())
attrs=[a for a in dir(model) if not a.startswith('_')]
info={'model':MODEL,'mlx_version':mx.__version__ if hasattr(mx,'__version__') else None,'model_type':str(type(model)),'attrs':attrs[:300]}
for name in ['model','lm_head','layers','args']:
    if hasattr(model,name):
        obj=getattr(model,name);info[f'attr_{name}_type']=str(type(obj))
        try: info[f'attr_{name}_shape']=list(obj.weight.shape) if hasattr(obj,'weight') else None
        except Exception as e: info[f'attr_{name}_shape_error']=repr(e)
if hasattr(model,'model'):
    inner=model.model
    info['inner_attrs']=[a for a in dir(inner) if not a.startswith('_')][:300]
    if hasattr(inner,'embed_tokens'):
        et=inner.embed_tokens;info['embed_tokens_type']=str(type(et));info['embed_weight_shape']=list(et.weight.shape);info['embed_weight_dtype']=str(et.weight.dtype)
ids=mx.array([[1,2,3,4]],dtype=mx.int32)
out=model(ids);mx.eval(out);info['forward_shape']=list(out.shape);info['forward_dtype']=str(out.dtype)
if hasattr(model,'model'):
    try:
        body=model.model(ids);mx.eval(body);info['inner_model_shape']=list(body.shape);info['inner_model_dtype']=str(body.dtype)
    except Exception as e: info['inner_model_error']=repr(e)
# Current MLX custom Metal API probe.
src='''
    uint i = thread_position_in_grid.x;
    if (i < 16) out[i] = inp[i] + T(1.0);
'''
try:
    k=mx.fast.metal_kernel(name='pb_probe',input_names=['inp'],output_names=['out'],source=src)
    x=mx.arange(16,dtype=mx.float32)
    y=k(inputs=[x],template=[('T',mx.float32)],grid=(16,1,1),threadgroup=(16,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype],init_value=0.0)[0]
    mx.eval(y);info['custom_kernel_ok']=bool(mx.all(y==x+1).item())
except Exception as e: info['custom_kernel_error']=repr(e)
# Verify SIMDgroup/threadgroup position built-ins accepted by MLX custom kernels.
simd_src='''
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    float v = (lane < 32) ? inp[row * 32 + lane] : 0.0f;
    float s = simd_sum(v);
    if (lane == 0) out[row] = s;
'''
try:
    sk=mx.fast.metal_kernel(name='pb_simd_probe',input_names=['inp'],output_names=['out'],source=simd_src)
    x=mx.ones((4,32),dtype=mx.float32)
    y=sk(inputs=[x],grid=(4*32,1,1),threadgroup=(32,1,1),output_shapes=[(4,)],output_dtypes=[mx.float32],init_value=0.0)[0]
    mx.eval(y);info['simd_kernel_values']=y.tolist();info['simd_kernel_ok']=bool(mx.all(y==32).item())
except Exception as e: info['simd_kernel_error']=repr(e)
for target,name in [(model,'model_call'),(getattr(model,'model',None),'inner_call')]:
    if target is not None:
        try: info[name+'_signature']=str(inspect.signature(target.__call__))
        except Exception as e: info[name+'_signature_error']=repr(e)
Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);p=Path('experiments/artifacts/proofbits_mlx_integration_probe.json');p.write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
