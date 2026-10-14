"""Component-owned training calibration and conservative workload/ROI matching.

Calibration is a separate hardware measurement, never a task-score benchmark
or auxiliary GPU load during an agent study. No agent/executor is patched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from .native_evidence import number
from .device_support import architecture, backend

SCHEMA = 'hwkg-calibration-v1'
V87_BUNDLED_SHA256 = '2f5a8949da17c424a6e7a317e0c4c1711280347df74741acd183666797329176'
V84_BUNDLED_SHA256 = '73e15fe8178a8cfb4d4d16be0dd87046a00b8e05d1aebea0703354b8d85571f5'
BUNDLED_SHA256 = '5912ee52bd8bd78f605432485284064c097320479feb677732f642641dfad7b5'


def calibration_path(device):
    if 'MLEVOLVE_HWKG_CALIBRATION_PATH' in os.environ:
        return os.environ['MLEVOLVE_HWKG_CALIBRATION_PATH']
    directory=Path(__file__).with_name('calibrations')
    priority=('rtx5080-v87-resnet50.json','rtx5080-v84-resnet50.json','rtx5080-resnet50-b64.json')
    paths=[directory/name for name in priority]
    paths += [p for p in sorted(directory.glob('*.json')) if p.name not in priority]
    for path in paths:
        # Discovery is metadata-only. select_record separately enforces owned,
        # non-symlink bounded input and exact GPU/runtime/workload matching.
        if path.is_symlink() or not path.is_file() or path.stat().st_size>2*1024**2: continue
        try: doc=json.loads(path.read_bytes())
        except (OSError,ValueError,TypeError): continue
        if not isinstance(doc,dict) or doc.get('schema')!=SCHEMA or not isinstance(doc.get('records'),list): continue
        for row in doc['records']:
            key=row.get('workload',{}) if isinstance(row,dict) else {}
            if (key.get('gpu_name')==device.name and key.get('capability')==[device.major,device.minor]
                    and (not architecture(device).startswith('gfx') or key.get('gpu_architecture')==architecture(device))):
                return path
    return None


def calibration_flags():
    import torch
    import torchvision
    return {'torchvision': str(torchvision.__version__),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}


def workload_key(device, runtime, spec):
    flags = {**runtime.get('calibration_flags', {}), **(spec.get('runtime_overrides') or {})}
    overrides = spec.get('runtime_overrides') or {}
    # Empirically verified in this pinned runtime using legacy flags alone.
    # Do not guess coupled semantics for another version or mixed API setters.
    if (runtime.get('torch') == '2.14.1+cu130'
            and type(overrides.get('matmul_allow_tf32')) is bool):
        flags['float32_matmul_precision'] = 'high' if overrides['matmul_allow_tf32'] else 'highest'
    key={'gpu_name': device.name, 'capability': [device.major, device.minor],
            'torch': runtime.get('torch'), 'cuda': runtime.get('cuda'),
            'runtime_flags': flags,
            **{k: spec.get(k) for k in ('model', 'optimizer', 'optimizer_options', 'weights',
                'classifier_classes', 'loss', 'batch', 'input_shape', 'forwards_per_update')}}
    if backend(device,runtime)=='rocm':
        key.update(hip=runtime.get('hip'),gpu_architecture=architecture(device))
    return key


def select_record(path, device, runtime, spec):
    if runtime.get('hardware_profile_identity_verified') is False:
        return None, 'exact hardware profile not identified; no variant/platform timing transfer'
    if not path: return None, 'no calibration configured'
    try:
        fd = os.open(Path(path), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > 2*1024**2:
                return None, 'calibration must be an owned bounded regular file'
            raw = stream.read(2*1024**2+1)
            if len(raw) > 2*1024**2: return None, 'calibration too large'
            doc = json.loads(raw)
    except (OSError, ValueError, TypeError, RecursionError): return None, 'calibration unavailable or invalid'
    if Path(path).absolute() == (Path(__file__).with_name('calibrations')/'rtx5080-resnet50-b64.json').absolute():
        if hashlib.sha256(raw).hexdigest() != BUNDLED_SHA256: return None, 'bundled calibration identity changed'
    if Path(path).absolute() == (Path(__file__).with_name('calibrations')/'rtx5080-v84-resnet50.json').absolute():
        if hashlib.sha256(raw).hexdigest() != V84_BUNDLED_SHA256: return None, 'bundled v8.4 calibration identity changed'
    if Path(path).absolute() == (Path(__file__).with_name('calibrations')/'rtx5080-v87-resnet50.json').absolute():
        if hashlib.sha256(raw).hexdigest() != V87_BUNDLED_SHA256: return None, 'bundled v8.7 calibration identity changed'
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA:
        return None, 'unrecognized calibration schema'
    if spec.get('training_scope') != 'full_model_source_screen':
        return None, 'parent training scope not matched'
    if spec.get('runtime_flags_mutated') and not spec.get('runtime_overrides'):
        return None, 'candidate backend settings not matched'
    if (type((spec.get('runtime_overrides') or {}).get('matmul_allow_tf32')) is bool
            and runtime.get('torch') != '2.14.1+cu130'):
        return None, 'literal TF32/precision coupling not verified for this runtime'
    key = workload_key(device, runtime, spec)
    if any(v is None for k,v in key.items() if not (k=='cuda' and backend(device,runtime)=='rocm')):
        return None, 'parent workload dimensions incomplete'
    records = doc.get('records')
    if not isinstance(records, list): return None, 'invalid calibration records'
    matches = [r for r in records if isinstance(r, dict) and r.get('workload') == key]
    if len(matches) != 1: return None, 'no unique exact host/runtime/workload calibration'
    row = {**matches[0], 'calibration_file_sha256': hashlib.sha256(raw).hexdigest()}
    profiles = row.get('profiles')
    if not isinstance(profiles, dict): return None, 'invalid calibration profiles'
    for name, p in profiles.items():
        if (name not in {d+'_'+l for d in ('fp32','fp16','bf16') for l in ('contiguous','channels_last')} or not isinstance(p, dict)
                or not number(p.get('step_seconds'), True) or not number(p.get('setup_seconds'))
                or type(p.get('repeats')) is not int or p['repeats'] < 3
                or type(p.get('normal_updates')) is not int or p['normal_updates'] < 20
                or p.get('finite_updates') is not True
                or p.get('finite_check_scope') != 'all_losses_initial_grad_final_parameters'):
            return None, 'calibration profile coverage/finite timings insufficient'
        if 'step_seconds_min' in p or 'step_seconds_max' in p:
            if (not number(p.get('step_seconds_min'), True) or not number(p.get('step_seconds_max'), True)
                    or not p['step_seconds_min'] <= p['step_seconds'] <= p['step_seconds_max']):
                return None, 'invalid repeat timing range'
    if not profiles: return None, 'empty calibration profiles'
    if row.get('quality_verified') is not False: return None, 'microcalibration cannot certify task quality'
    return row, 'exact host/runtime/workload microcalibration; task quality and full-pipeline gain unverified'


def choose_trial(record, current, updates, overhead_seconds=0., blocked_profiles=()):
    if (not record or type(updates) is not int or updates < 20
            or not number(overhead_seconds)): return None
    profiles = record['profiles']; baseline = profiles.get(current)
    if not baseline: return None
    candidates = []
    for name, p in profiles.items():
        if name == current or name in blocked_profiles: continue
        # One change only; do not bundle dtype and layout into an attribution.
        old_dtype, old_layout = current.split('_', 1)
        new_dtype, new_layout = name.split('_', 1)
        if (old_dtype != new_dtype) + (old_layout != new_layout) != 1:
            continue
        saved = updates*(baseline['step_seconds']-p['step_seconds'])
        extra_setup = max(0., p['setup_seconds']-baseline['setup_seconds'])
        net = saved-extra_setup-overhead_seconds
        has_range = all(k in profile for profile in (baseline,p) for k in ('step_seconds_min','step_seconds_max'))
        conservative_net = (updates*(baseline['step_seconds_min']-p['step_seconds_max'])
                            -extra_setup-overhead_seconds) if has_range else net
        if conservative_net > 0 and p['step_seconds'] <= .95*baseline['step_seconds']:
            candidates.append({'profile': name, 'expected_update_savings_seconds': saved,
                'additional_setup_seconds': extra_setup, 'declared_generation_overhead_seconds': overhead_seconds,
                'change': 'layout' if old_layout != new_layout else 'precision',
                'estimated_net_seconds': net, 'step_speedup': baseline['step_seconds']/p['step_seconds'],
                'repeat_spread_net_seconds': conservative_net if has_range else None,
                'uncertainty_basis':'observed fastest baseline vs slowest target repeat; not statistical CI' if has_range else 'repeat spread unavailable; median-only projection',
                'quality_verified': False, 'end_to_end_verified': False})
    return max(candidates, key=lambda p:p['estimated_net_seconds']) if candidates else None


def calibrate(images_path, csv_path, weights_path, output, batch=64, steps=25, warmup=5, repeats=3,
              cudnn_benchmark=False, label_smoothing=0., lr=.001, weight_decay=.0001, matmul_tf32=None):
    import numpy as np
    import pandas as pd
    import statistics
    import torch
    import torch.nn.functional as F
    import torchvision
    from torchvision.models import resnet50
    if steps < 20 or repeats < 3 or warmup < 1 or batch < 2: raise ValueError('Insufficient calibration coverage')
    if (not number(label_smoothing) or label_smoothing > 1 or not number(lr, True)
            or not number(weight_decay) or (matmul_tf32 is not None and type(matmul_tf32) is not bool)):
        raise ValueError('Invalid exact calibration recipe')
    if torch.cuda.device_count() != 1: raise ValueError('Exactly one visible GPU required')
    device = torch.cuda.get_device_properties(0)
    if architecture(device)=='unknown': raise ValueError('Unrecognized GPU architecture')
    output = Path(output)
    if output.exists(): raise ValueError('Fresh calibration file required')
    if Path(csv_path).name != 'train.csv' or Path(images_path).name != 'train_images.npy':
        raise ValueError('Use the public training files only')
    frame = pd.read_csv(csv_path)
    if not {'image_row', 'target'} <= set(frame.columns): raise ValueError('Public training CSV required')
    selected = frame.iloc[:batch*(steps+warmup)]
    if len(selected) != batch*(steps+warmup): raise ValueError('Insufficient real training rows')
    array = np.load(images_path, mmap_mode='r', allow_pickle=False)
    if array.shape[1:] != (3, 32, 32): raise ValueError('Expected public CIFAR RGB training images')
    if (selected.image_row.dtype.kind not in 'iu' or selected.target.dtype.kind not in 'iu'
            or not selected.image_row.between(0,len(array)-1).all()
            or not selected.target.between(0,99).all()): raise ValueError('Invalid public training rows/classes')
    x = torch.from_numpy(np.array(array[selected.image_row.to_numpy()], copy=True)).float()/255
    x = F.interpolate(x, size=(128,128), mode='bilinear', align_corners=False)
    x = ((x-torch.tensor([.485,.456,.406])[None,:,None,None])/
         torch.tensor([.229,.224,.225])[None,:,None,None]).pin_memory()
    y = torch.from_numpy(selected.target.to_numpy().copy()).long().pin_memory()
    from torchvision.models import ResNet50_Weights
    weight_sha = hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()
    expected_prefix = Path(ResNet50_Weights.IMAGENET1K_V2.url).stem.split('-')[-1]
    if not weight_sha.startswith(expected_prefix): raise ValueError('Expected official ResNet50 V2 weights')
    weights = torch.load(weights_path, map_location='cpu', weights_only=True)
    profiles = {}; raw = []
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if matmul_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    # Rotating order reduces a systematic first-profile warmup/thermal bias.
    try: bf16=torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError: bf16=False
    dtypes=('fp32','fp16','bf16') if bf16 else ('fp32','fp16')
    names = [d+'_'+l for d in dtypes for l in ('contiguous','channels_last')]
    for repeat in range(repeats):
        for name in names[repeat%len(names):]+names[:repeat%len(names)]:
            torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            model = resnet50(weights=None);model.load_state_dict(weights)
            model.fc = torch.nn.Linear(model.fc.in_features, 100)
            fmt = torch.channels_last if name.endswith('channels_last') else torch.contiguous_format
            model = model.cuda().to(memory_format=fmt);model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scaler = torch.amp.GradScaler('cuda', enabled=name.startswith('fp16'), init_scale=1024.)
            torch.cuda.synchronize(); setup = time.perf_counter()-t0
            start = None; finite_losses = []
            for i in range(steps+warmup):
                if i == warmup:
                    torch.cuda.synchronize(); start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                xb=x[i*batch:(i+1)*batch].cuda(non_blocking=True).contiguous(memory_format=fmt)
                yb=y[i*batch:(i+1)*batch].cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16 if name.startswith('fp16') else torch.bfloat16,
                                    enabled=not name.startswith('fp32')):
                    loss=F.cross_entropy(model(xb),yb,label_smoothing=label_smoothing)
                scaler.scale(loss).backward()
                if i == 0: scaler.unscale_(optimizer)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer);scaler.update()
                if scaler.get_scale() < previous_scale: raise ValueError('Skipped FP16 update; calibration invalid')
                finite_losses.append(torch.isfinite(loss.detach()))
                if i == 0:
                    assert torch.isfinite(loss) and all(torch.isfinite(p).all() for p in model.parameters())
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            assert torch.stack(finite_losses).all() and all(torch.isfinite(p).all() for p in model.parameters())
            raw.append({'profile':name,'repeat':repeat,'step_seconds':elapsed/steps,'setup_seconds':setup,
                        'examples_per_second':batch*steps/elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
            del model,optimizer,scaler,xb,yb,loss,finite_losses
    for name in names:
        rows=[r for r in raw if r['profile']==name]
        profiles[name]={'step_seconds':statistics.median(r['step_seconds'] for r in rows),
            'step_seconds_min':min(r['step_seconds'] for r in rows),
            'step_seconds_max':max(r['step_seconds'] for r in rows),
            'setup_seconds':statistics.median(r['setup_seconds'] for r in rows),
            'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows),
            'normal_updates':steps,'repeats':repeats,'finite_updates':True,
            'finite_check_scope':'all_losses_initial_grad_final_parameters'}
    spec={'model':'torchvision.resnet50','optimizer':'AdamW','batch':batch,'input_shape':[3,128,128],'forwards_per_update':1,
          'weights':'ResNet50_Weights.IMAGENET1K_V2','classifier_classes':100,
          'loss':'cross_entropy_default' if label_smoothing == 0 else {'kind':'cross_entropy','label_smoothing':float(label_smoothing),'reduction':'mean'},
          'optimizer_options':{'lr':lr,'betas':[.9,.999],'eps':1e-8,'weight_decay':weight_decay,
              'amsgrad':False,'maximize':False,'foreach':None,'capturable':False,'differentiable':False,'fused':None}}
    result={'schema':SCHEMA,'records':[{'workload':workload_key(device,{'torch':str(torch.__version__),'cuda':torch.version.cuda,
                                                                       'hip':getattr(torch.version,'hip',None),
                                                                       'calibration_flags':calibration_flags()},spec),
        'profiles':profiles,'quality_verified':False,'scope':'real public train minibatches, full-model update/H2D; no augmentation or task-score evaluation',
        'weights_sha256':weight_sha}], 'raw_trials':raw,
        'public_training_csv_sha256':hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        'torchvision':torchvision.__version__,'measurement_time':time.time()}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Separate HWKG public-data calibration; no agent/search/API calls')
    for key in ['images','csv','weights','output']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--cudnn-benchmark',action='store_true')
    parser.add_argument('--label-smoothing',type=float,default=0.)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--weight-decay',type=float,default=.0001)
    parser.add_argument('--matmul-tf32',action='store_true',default=None)
    args=parser.parse_args()
    result=calibrate(args.images,args.csv,args.weights,args.output,args.batch,cudnn_benchmark=args.cudnn_benchmark,
        label_smoothing=args.label_smoothing,lr=args.lr,weight_decay=args.weight_decay,matmul_tf32=args.matmul_tf32)
    print(json.dumps({'output':args.output,'profiles':result['records'][0]['profiles'],'quality_verified':False}))
