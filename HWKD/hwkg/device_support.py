"""Backend and capability routing shared by component-owned GPU advice.

These are eligibility checks, not kernel execution or speed measurements.
"""
import re

AMP_TRIAL = 'feat:runtime_amp_trial'
FUSED_TRIAL = 'feat:runtime_fused_adamw_trial'


def backend(device, runtime=None):
    runtime = runtime or {}
    gfx = getattr(device, 'gcnArchName', None)
    if runtime.get('hip') or (isinstance(gfx, str) and gfx.startswith('gfx')):
        return 'rocm'
    return runtime.get('backend', 'cuda')


def architecture(device):
    gfx = getattr(device, 'gcnArchName', '')
    if isinstance(gfx, str) and gfx.startswith('gfx'):
        match = re.fullmatch(r'(gfx[0-9a-f]+)(?::[^\s]*)?', gfx)
        return match[1] if match else 'unknown'
    major, minor = getattr(device, 'major', None), getattr(device, 'minor', None)
    return f'sm_{major}{minor}' if type(major) is int and type(minor) is int else 'unknown'


def capability(device):
    target = architecture(device)
    return target if target.startswith('gfx') else f'{device.major}.{device.minor}'


def runtime_label(device, runtime):
    family = backend(device, runtime)
    return ('ROCm' + str(runtime.get('hip', 'unknown')) if family == 'rocm'
            else 'CUDA' + str(runtime.get('cuda', 'unknown')))


def supported(edges, feature):
    props = edges.get(feature, {}).get('properties', {})
    return props.get('verified') is True and props.get('support_level') == 'supported'


def options(device, edges, runtime):
    """Intersect graph permission with this installed backend's capability facts."""
    family = backend(device, runtime)
    amp_edge = supported(edges, 'feat:amp') or supported(edges, AMP_TRIAL)
    amp = amp_edge and runtime.get('amp_api', True) is True
    # NVIDIA requires its native BF16 hardware edge; emulation is not eligibility.
    bf16_edge = (supported(edges, 'feat:bf16') if family == 'cuda'
                 else supported(edges, AMP_TRIAL))
    bf16 = amp and bf16_edge and runtime.get('bf16_supported') is True
    fp16 = amp and runtime.get('grad_scaler_api') is True
    fused_edge = (supported(edges, FUSED_TRIAL)
                  or supported(edges, 'feat:optimizer_fused_adamw'))
    fused = (fused_edge and runtime.get('adamw_fused_api') is True
             and runtime.get('adamw_fused_kernel', True) is True)
    return {'backend': family, 'target': architecture(device), 'amp': amp,
            'bf16': bf16, 'fp16': fp16, 'fused': fused,
            'amp_feature': 'feat:amp' if supported(edges, 'feat:amp') else AMP_TRIAL,
            'fused_feature': FUSED_TRIAL if supported(edges, FUSED_TRIAL)
                             else 'feat:optimizer_fused_adamw'}
