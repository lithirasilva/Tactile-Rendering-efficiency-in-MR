import os
import json
import time
from typing import List, Dict, Any

import numpy as np

from vibtac_physics_scenarios import VibTacPhysicsGenerator
from haptic_renderer import HapticRenderer, ProbeState


WORKDIR = os.path.dirname(__file__)
PERF_DIR = os.path.join(WORKDIR, 'logs', 'performance')
EXP_DIR = os.path.join(WORKDIR, 'logs', 'experiments')


def load_json_files(folder: str, suffix: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not os.path.isdir(folder):
        return results
    for name in sorted(os.listdir(folder)):
        if not name.endswith(suffix):
            continue
        path = os.path.join(folder, name)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
        except Exception:
            continue
    return results


def summarize_predict_perf(perf: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        'files': len(perf),
        'avg_inference_ms': None,
        'avg_throughput_rps': None,
        'device': None,
        'examples': []
    }

    if not perf:
        return out

    infs = []
    thr = []
    devices = []
    for p in perf:
        try:
            infs.append(float(p.get('inference_time', {}).get('avg_ms', 0)))
        except Exception:
            pass
        try:
            thr.append(float(p.get('throughput_rps', 0)))
        except Exception:
            pass
        server = p.get('server_info', {})
        if server:
            devices.append(server.get('device'))

        # up to 3 example predictions
        for item in p.get('predictions', [])[:3]:
            out['examples'].append({
                'input': item.get('input'),
                'texture': item.get('output', {}).get('texture'),
                'confidence': item.get('output', {}).get('confidence'),
                'force_n': item.get('output', {}).get('force'),
                'physics_force_n': item.get('output', {}).get('physics_force'),
                'inference_ms': item.get('inference_ms'),
                'latency_ms': item.get('latency_ms')
            })

    out['avg_inference_ms'] = float(np.mean(infs)) if infs else None
    out['avg_throughput_rps'] = float(np.mean(thr)) if thr else None
    out['device'] = devices[-1] if devices else None
    return out


def summarize_haptic_perf(perf: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        'files': len(perf),
        'median_ms': None,
        'p95_ms': None,
        'throughput_rps': None,
        'device': None
    }
    if not perf:
        return out
    out['median_ms'] = float(np.mean([p.get('median_ms', 0) for p in perf]))
    out['p95_ms'] = float(np.mean([p.get('p95_ms', 0) for p in perf]))
    out['throughput_rps'] = float(np.mean([p.get('throughput_rps', 0) for p in perf]))
    last = perf[-1]
    out['device'] = last.get('server_info', {}).get('device')
    return out


def summarize_experiments(exps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for e in exps:
        summary.append({
            'texture': e.get('material_properties', {}).get('texture_name'),
            'scenario': e.get('config', {}).get('scenario'),
            'signal_rms': e.get('signal_rms'),
            'predicted_texture': e.get('predicted_texture_name'),
            'texture_confidence': e.get('texture_confidence'),
            'predicted_force_n': e.get('predicted_force'),
            'physics_force_n': e.get('physics_force'),
            'inference_time_ms': e.get('inference_time_ms'),
            'sound': e.get('sound_prediction', {})
        })
    return summary


def haptic_samples() -> List[Dict[str, Any]]:
    gen = VibTacPhysicsGenerator()
    renderer = HapticRenderer()

    def run(texture_id: int, label: str, position, velocity):
        mat = gen.get_material(texture_id)
        probe = ProbeState(position=np.array(position), velocity=np.array(velocity), last_contact_time=0.1)
        start = time.time()
        result = renderer.render_force_field(probe, texture_id, {
            'stiffness': mat.stiffness,
            'friction': mat.friction,
            'roughness': mat.roughness,
            'damping': mat.damping,
        })
        render_time_ms = (time.time() - start) * 1000.0
        return {
            'material': mat.texture_name,
            'interaction': label,
            'force_magnitude_n': result['force_magnitude'],
            'render_time_ms': render_time_ms,
            'force_vector': result['force_vector'],
            'vibration_amplitude': result['vibration_amplitude']
        }

    return [
        run(1, 'slide', [0.0, 0.0, -0.002], [0.20, 0.0, 0.0]),   # Aluminum film + slide
        run(0, 'tap',   [0.0, 0.0, -0.002], [0.00, 0.0, 0.0]),   # Fabric-1 + tap
        run(11, 'press', [0.0, 0.0, -0.002], [0.05, 0.0, 0.0])   # Rubber + press
    ]


def main():
    # Load performance logs
    predict_perf = load_json_files(PERF_DIR, 'predict_perf.json')
    haptic_perf = load_json_files(PERF_DIR, 'haptic_perf.json')

    # Summaries
    predict_summary = summarize_predict_perf(predict_perf)
    haptic_summary = summarize_haptic_perf(haptic_perf)

    # Experiments
    experiments = load_json_files(EXP_DIR, '.json')
    exp_summary = summarize_experiments(experiments)

    # Haptic renderer samples
    haptic = haptic_samples()

    report = {
        'predict_performance': predict_summary,
        'haptic_performance': haptic_summary,
        'experiment_examples': exp_summary[:3],  # show top 3
        'haptic_samples': haptic
    }

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
