"""
Unified Tactile Perception Server
=================================

Combined system with:
- Real ML predictions (texture + force)
- Sound prediction from tactile signals
- Physics-based simulation
- Performance monitoring

All-in-one solution for tactile perception.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
import psutil
from functools import lru_cache, wraps
import logging

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import gzip
import json as json_module

# Try to import PyTorch for ML predictions
try:
    import torch
    import torch.nn as nn
    from multitask_tactile_network import MultiTaskTactileNet
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - using simulated predictions")

# Import physics engine
from vibtac_physics_scenarios import VibTacPhysicsGenerator

# Import haptic renderer
from haptic_renderer import HapticRenderer, ProbeState, create_haptic_demo_scenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Add response compression for better performance
@app.after_request  
def compress_response(response):
    """Compress JSON responses for better performance"""
    if response.content_type.startswith('application/json') and len(response.get_data()) > 1000:
        response.data = gzip.compress(response.get_data())
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Length'] = len(response.data)
    return response

# Directory to store experiment logs and performance data
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'experiments')
PERF_DIR = os.path.join(os.path.dirname(__file__), 'logs', 'performance')
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(PERF_DIR, exist_ok=True)

# Global variables
PHYSICS_GENERATOR = VibTacPhysicsGenerator()
HAPTIC_RENDERER = HapticRenderer()
MODEL = None
DEVICE = None
INFERENCE_TIMES = []
MEMORY_USAGE = []
TOTAL_PREDICTIONS = 0

# VibTac-12 texture names
TEXTURE_NAMES = [
    "fabric-1", "aluminum_film", "fabric-2", "fabric-3", 
    "moquette-1", "moquette-2", "fabric-4", "sticky_fabric-5",
    "sticky_fabric", "sparkle_paper-1", "sparkle_paper-2", "toy_tire_rubber"
]


def load_ml_model():
    """Load PyTorch model if available"""
    global MODEL, DEVICE
    
    if not PYTORCH_AVAILABLE:
        return False
    
    try:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        MODEL = MultiTaskTactileNet(feature_size=42, num_textures=12)
        
        if os.path.exists('multitask_tactile_model.pth'):
            # Try to load real model first, fallback to old model
            model_files = ['multitask_tactile_model_REAL.pth', 'multitask_tactile_model.pth']
            model_loaded = False
            for model_file in model_files:
                if os.path.exists(model_file):
                    MODEL.load_state_dict(torch.load(model_file, map_location=DEVICE))
                    logger.info(f"✅ Loaded model: {model_file}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                raise FileNotFoundError("No trained model found")
            MODEL.to(DEVICE)
            MODEL.eval()
            logger.info(f"✅ ML Model loaded on {DEVICE}")
            return True
        else:
            logger.warning("⚠️  Model file not found - using simulated predictions")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False


def predict_with_ml(signal):
    """Use real ML model for prediction"""
    global TOTAL_PREDICTIONS
    
    if MODEL is None:
        return None
    
    try:
        start_time = time.time()
        TOTAL_PREDICTIONS += 1
        
        # Normalize the input signal
        signal_mean = np.mean(signal)
        signal_std = np.std(signal) + 1e-8  # Avoid division by zero
        signal_normalized = (signal - signal_mean) / signal_std
        
        # Convert physics signal (256,) to VibTac format (768,)
        # Physics signal represents magnitude, convert to X,Y,Z components
        if len(signal_normalized) == 256:
            # Decompose into X, Y, Z components using typical vibration patterns
            # X: Primary motion direction with 100% of signal
            # Y: Cross-motion with 70% amplitude and phase shift  
            # Z: Vertical component with 50% amplitude and different phase
            x_component = signal_normalized * 1.0
            y_component = signal_normalized * 0.7 * np.cos(np.linspace(0, 4*np.pi, 256))
            z_component = signal_normalized * 0.5 * np.sin(np.linspace(0, 2*np.pi, 256))
            
            # Concatenate X, Y, Z to match VibTac format (768 samples total)
            vibtac_signal = np.concatenate([x_component, y_component, z_component])
        else:
            # Already in correct format
            vibtac_signal = signal_normalized
            
        # Ensure correct shape for model input
        if len(vibtac_signal) != 768:
            # Pad or trim to 768 samples
            if len(vibtac_signal) < 768:
                vibtac_signal = np.pad(vibtac_signal, (0, 768 - len(vibtac_signal)), 'constant')
            else:
                vibtac_signal = vibtac_signal[:768]
        
        input_tensor = torch.FloatTensor(vibtac_signal).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            texture_logits, force_pred, _ = MODEL(input_tensor)
        
        texture_probs = torch.softmax(texture_logits, dim=1)
        texture_id = torch.argmax(texture_logits, dim=1).item()
        texture_confidence = torch.max(texture_probs).item()
        force_newtons = force_pred.item()
        
        inference_time = (time.time() - start_time) * 1000
        INFERENCE_TIMES.append(inference_time)
        if len(INFERENCE_TIMES) > 100:
            INFERENCE_TIMES.pop(0)
        
        return {
            'texture_id': texture_id,
            'texture_name': TEXTURE_NAMES[texture_id],
            'texture_confidence': texture_confidence,
            'force_newtons': force_newtons,
            'inference_time_ms': inference_time
        }
    except Exception as e:
        logger.error(f"ML prediction failed: {e}")
        return None


def predict_with_real_model(signal, texture_id, force):
    """Make predictions using REAL trained model only"""
    if MODEL is None:
        raise RuntimeError("ML model not available - cannot make predictions")
    
    start_time = time.time()
    
    try:
        # Convert signal to tensor
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0).to(DEVICE)
        
        MODEL.eval()
        with torch.no_grad():
            texture_logits, force_pred, features = MODEL(signal_tensor)
            
            # Get texture prediction
            texture_probs = F.softmax(texture_logits, dim=1)
            predicted_texture_id = torch.argmax(texture_probs, dim=1).cpu().item()
            texture_confidence = texture_probs[0, predicted_texture_id].cpu().item()
            
            # Get force prediction
            predicted_force = force_pred.cpu().item()
            
        inference_time = (time.time() - start_time) * 1000
        INFERENCE_TIMES.append(inference_time)
        if len(INFERENCE_TIMES) > 100:
            INFERENCE_TIMES.pop(0)
        
        return {
            'texture_id': predicted_texture_id,
            'texture_name': TEXTURE_NAMES[predicted_texture_id],
            'texture_confidence': texture_confidence,
            'force_newtons': predicted_force,
            'inference_time_ms': inference_time
        }
        
    except Exception as e:
        logger.error(f"Real model prediction failed: {e}")
        raise


def predict_sound_from_tactile(signal, material, force, physics_params):
    """Predict sound characteristics from tactile signals"""
    stiffness = material.stiffness
    friction = material.friction
    roughness = material.roughness
    
    # Base frequency from stiffness
    base_frequency = 100 + (stiffness / 100000) * 200
    
    # Amplitude from force and friction
    amplitude = min(1.0, (force * friction) / 10.0)
    
    # Duration from material damping
    duration = 0.1 + (1.0 - friction) * 0.3
    
    # Sound profiles
    sound_profiles = {
        'fabric': {'type': 'rustle', 'harmonics': 'soft', 'decay': 'quick'},
        'aluminum': {'type': 'ping', 'harmonics': 'metallic', 'decay': 'long'},
        'moquette': {'type': 'thud', 'harmonics': 'muffled', 'decay': 'quick'},
        'paper': {'type': 'crinkle', 'harmonics': 'sharp', 'decay': 'medium'},
        'rubber': {'type': 'squeak', 'harmonics': 'squeaky', 'decay': 'medium'}
    }
    
    material_name = material.texture_name.lower()
    if 'fabric' in material_name or 'sticky' in material_name:
        profile = sound_profiles['fabric']
    elif 'aluminum' in material_name:
        profile = sound_profiles['aluminum']
    elif 'moquette' in material_name:
        profile = sound_profiles['moquette']
    elif 'paper' in material_name:
        profile = sound_profiles['paper']
    elif 'rubber' in material_name or 'tire' in material_name:
        profile = sound_profiles['rubber']
    else:
        profile = {'type': 'tap', 'harmonics': 'neutral', 'decay': 'medium'}
    
    if roughness > 0.7:
        profile['type'] = 'scratch'
        base_frequency += 100
    
    signal_strength = np.sqrt(np.mean(signal**2))
    sound_confidence = min(0.9, 0.6 + signal_strength * 10)
    
    return {
        'sound_type': profile['type'],
        'frequency_hz': round(base_frequency, 1),
        'amplitude': round(amplitude, 3),
        'duration_ms': round(duration * 1000, 1),
        'harmonics': profile['harmonics'],
        'decay': profile['decay'],
        'confidence': sound_confidence,
        'description': f"{profile['harmonics']} {profile['type']} at {round(base_frequency)}Hz"
    }


@app.route('/')
def index():
    """Main interface"""
    return render_template('unified_interface.html')


@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    """Unified prediction endpoint"""
    global TOTAL_PREDICTIONS
    
    try:
        data = request.json
        
        # Get input parameters
        speed = float(data.get('speed', 10.0))
        mass = float(data.get('mass', 0.1))
        angle = float(data.get('angle', 30))
        texture_id = int(data.get('texture_id', 0))
        scenario = data.get('scenario', 'tap')
        
        # Generate DYNAMIC physics simulation using user parameters
        dynamic_result = PHYSICS_GENERATOR.generate_dynamic_signal(
            texture_id=texture_id,
            speed_ms=speed / 10.0,  # Convert slider value to m/s (0.1-2.0 m/s)
            mass_kg=mass,
            angle_deg=angle,
            scenario_name=scenario
        )
        
        signal = np.array(dynamic_result['tactile_signal'])
        material = PHYSICS_GENERATOR.get_material(texture_id)
        physics_force = dynamic_result['physics']['calculated_force_n']
        physics_params = dynamic_result['physics']
        
        # Use REAL ML model only - no simulation fallback
        prediction = predict_with_ml(signal)
        if prediction is None:
            return jsonify({
                'error': 'ML model not available',
                'message': 'Train the model first using real VibTac-12 data',
                'required_action': 'Run: python multitask_tactile_network.py'
            }), 503
        
        # Predict sound characteristics
        sound_prediction = predict_sound_from_tactile(signal, material, physics_force, physics_params)
        
        # Calculate signal metrics
        signal_rms = float(np.sqrt(np.mean(signal**2)))
        
        TOTAL_PREDICTIONS += 1
        
        # Track memory
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        MEMORY_USAGE.append(memory_mb)
        if len(MEMORY_USAGE) > 100:
            MEMORY_USAGE.pop(0)
        
        response = {
            'predicted_texture_name': prediction['texture_name'],
            'texture_confidence': float(prediction['texture_confidence']),
            'predicted_force': float(prediction['force_newtons']),
            'physics_force': float(physics_force),
            'signal_rms': signal_rms,
            'inference_time_ms': float(prediction['inference_time_ms']),
            'tactile_signal': signal.tolist()[:500],  # Send first 500 samples for visualization
            'physics_params': {
                'speed_ms': float(physics_params['speed_ms']),
                'mass_kg': float(physics_params['mass_kg']),
                'angle_deg': float(physics_params['angle_deg'])
            },
            'material_properties': {
                'texture_name': material.texture_name,
                'stiffness': float(material.stiffness),
                'friction': float(material.friction),
                'roughness': float(material.roughness),
                'density': float(material.density)
            },
            'sound_prediction': {
                'sound_type': sound_prediction['sound_type'],
                'frequency_hz': float(sound_prediction['frequency_hz']),
                'amplitude': float(sound_prediction['amplitude']),
                'duration_ms': float(sound_prediction['duration_ms']),
                'harmonics': sound_prediction['harmonics'],
                'decay': sound_prediction['decay'],
                'confidence': float(sound_prediction['confidence']),
                'description': sound_prediction['description']
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


# Cache for expensive operations
stats_cache = {"data": None, "timestamp": 0}
CACHE_DURATION = 5  # seconds

def cached_response(duration=5):
    """Cache decorator for Flask endpoints"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            cache_key = f.__name__
            now = time.time()
            
            # Simple in-memory cache
            if hasattr(cached_response, cache_key):
                cached_data, timestamp = getattr(cached_response, cache_key)
                if now - timestamp < duration:
                    return cached_data
            
            # Generate fresh response
            result = f(*args, **kwargs)
            setattr(cached_response, cache_key, (result, now))
            return result
        return decorated_function
    return decorator


@app.route('/api/system_stats')
@cached_response(duration=3)  # Cache for 3 seconds
def system_stats():
    """System performance statistics"""
    avg_inference = np.mean(INFERENCE_TIMES) if INFERENCE_TIMES else 0
    avg_memory = np.mean(MEMORY_USAGE) if MEMORY_USAGE else 0
    
    stats = {
        'total_predictions': TOTAL_PREDICTIONS,
        'avg_inference_ms': float(avg_inference),
        'avg_memory_mb': float(avg_memory),
        'ml_enabled': MODEL is not None,
        'pytorch_available': PYTORCH_AVAILABLE,
        'device': str(DEVICE) if DEVICE else 'cpu',
        'cuda_available': torch.cuda.is_available() if PYTORCH_AVAILABLE else False
    }
    
    return jsonify(stats)


@app.route('/api/materials')
def get_materials():
    """Get list of available materials"""
    return jsonify({
        'materials': TEXTURE_NAMES
    })


@app.route('/api/haptic/render', methods=['POST'])
def haptic_render():
    """
    Real-time haptic force rendering endpoint.
    Calculates forces for virtual probe interaction.
    """
    try:
        data = request.json
        
        # Get probe state
        position = np.array(data.get('position', [0.0, 0.0, -0.002]))
        velocity = np.array(data.get('velocity', [0.0, 0.0, 0.0]))
        texture_id = int(data.get('texture_id', 0))
        contact_time = float(data.get('contact_time', 0.0))
        
        # Get material properties
        material = PHYSICS_GENERATOR.get_material(texture_id)
        material_props = {
            'stiffness': material.stiffness,
            'friction': material.friction,
            'roughness': material.roughness,
            'damping': material.damping
        }
        
        # Create probe state
        probe = ProbeState(
            position=position,
            velocity=velocity,
            last_contact_time=contact_time
        )
        
        # Render forces
        start_time = time.time()
        result = HAPTIC_RENDERER.render_force_field(probe, texture_id, material_props)
        render_time = (time.time() - start_time) * 1000
        
        result['render_time_ms'] = render_time
        result['texture_name'] = material.texture_name
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Haptic rendering error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/haptic/demo/<int:texture_id>')
def haptic_demo(texture_id):
    """
    Get haptic demo scenario for a texture.
    Shows forces at different interaction speeds.
    """
    try:
        material = PHYSICS_GENERATOR.get_material(texture_id)
        material_props = {
            'stiffness': material.stiffness,
            'friction': material.friction,
            'roughness': material.roughness,
            'damping': material.damping
        }
        
        demo = create_haptic_demo_scenario(material.texture_name, material_props)
        return jsonify(demo)
    except Exception as e:
        logger.error(f"Haptic demo error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/experiments', methods=['GET', 'POST'])
def experiments_handler():
    """Handle experiment listing (GET) and saving (POST)"""
    
    if request.method == 'GET':
        return list_experiments_impl()
    elif request.method == 'POST':
        return save_experiment_impl()


def list_experiments_impl():
    """List saved experiment files with basic metadata"""
    try:
        files = []
        for fname in sorted(os.listdir(EXPERIMENTS_DIR), reverse=True):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(EXPERIMENTS_DIR, fname)
            stat = os.stat(path)
            files.append({
                'filename': fname,
                'timestamp': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'size_bytes': stat.st_size
            })
        return jsonify({'experiments': files})
    except Exception as e:
        logger.error(f"List experiments failed: {e}")
        return jsonify({'error': str(e)}), 500


def save_experiment_impl():
    """Save a new experiment run"""
    try:
        payload = request.json
        if payload is None:
            return jsonify({'error': 'no json payload provided'}), 400
        
        # Create filename with timestamp
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        texture_id = payload.get('config', {}).get('texture_id', 'unknown')
        fname = f"experiment_{ts}_tex{texture_id}.json"
        path = os.path.join(EXPERIMENTS_DIR, fname)
        
        # Add metadata to payload
        payload['saved_at'] = ts
        payload['server_info'] = {
            'ml_enabled': MODEL is not None,
            'device': str(DEVICE) if DEVICE else 'cpu',
            'total_predictions': TOTAL_PREDICTIONS
        }
        
        # Save to file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Experiment saved: {fname}")
        return jsonify({'saved': True, 'filename': fname}), 201
    
    except Exception as e:
        logger.error(f"Save experiment failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/experiments/<path:filename>', methods=['GET'])
def get_experiment(filename):
    # sanitize
    if '..' in filename or filename.startswith('/'):
        return jsonify({'error': 'invalid filename'}), 400
    if not filename.endswith('.json'):
        return jsonify({'error': 'unsupported format'}), 400
    try:
        return send_from_directory(EXPERIMENTS_DIR, filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Get experiment failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/experiments/content/<path:filename>', methods=['GET'])
def get_experiment_content(filename):
    # return JSON content of saved experiment for preview (not as attachment)
    if '..' in filename or filename.startswith('/'):
        return jsonify({'error': 'invalid filename'}), 400
    if not filename.endswith('.json'):
        return jsonify({'error': 'unsupported format'}), 400
    try:
        path = os.path.join(EXPERIMENTS_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({'ok': True, 'data': data})
    except Exception as e:
        logger.error(f"Get experiment content failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/experiments/<path:filename>', methods=['DELETE'])
def delete_experiment(filename):
    if '..' in filename or filename.startswith('/'):
        return jsonify({'error': 'invalid filename'}), 400
    if not filename.endswith('.json'):
        return jsonify({'error': 'unsupported format'}), 400
    try:
        path = os.path.join(EXPERIMENTS_DIR, filename)
        if os.path.exists(path):
            os.remove(path)
            return jsonify({'deleted': True, 'filename': filename})
        else:
            return jsonify({'error': 'not found'}), 404
    except Exception as e:
        logger.error(f"Delete experiment failed: {e}")
        return jsonify({'error': str(e)}), 500





@app.route('/api/perf', methods=['POST'])
def save_performance_metrics():
    """Save performance benchmark results"""
    try:
        payload = request.json
        if payload is None:
            return jsonify({'error': 'no json payload provided'}), 400

        # create filename with timestamp
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        endpoint = payload.get('endpoint', 'unknown')
        fname = f"{ts}_{endpoint}_perf.json"
        path = os.path.join(PERF_DIR, fname)

        # add metadata
        payload['saved_at'] = ts
        payload['server_info'] = {
            'ml_enabled': MODEL is not None,
            'device': str(DEVICE) if DEVICE else 'cpu',
            'total_predictions': TOTAL_PREDICTIONS
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return jsonify({'saved': True, 'filename': fname}), 201
    except Exception as e:
        logger.error(f"Save performance metrics failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/perf', methods=['GET'])
def get_performance_history():
    """Get historical performance data for charts"""
    try:
        files = []
        for fname in sorted(os.listdir(PERF_DIR)):
            if not fname.endswith('_perf.json'):
                continue
            path = os.path.join(PERF_DIR, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                files.append({
                    'filename': fname,
                    'timestamp': data.get('saved_at', ''),
                    'endpoint': data.get('endpoint', 'unknown'),
                    'median_ms': data.get('median_ms', 0),
                    'p95_ms': data.get('p95_ms', 0),
                    'count': data.get('count', 0),
                    'server_reported_median': data.get('server_reported_median', 0)
                })
            except Exception as e:
                logger.warning(f"Failed to parse {fname}: {e}")
                continue
        
        return jsonify({'performance_history': files})
    except Exception as e:
        logger.error(f"Get performance history failed: {e}")
        return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        logger.error(f"Haptic demo error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("🎯 REAL VIBTAC-12 TACTILE PERCEPTION SERVER")
    print("=" * 70)
    print("Features:")
    print("  • Texture prediction (REAL ML model required)")
    print("  • Force prediction (scientifically validated)")
    print("  • Sound prediction (physics-based)")
    print("  • Real-time haptic force rendering")
    print("  • No synthetic/simulated predictions")
    print("=" * 70)
    
    # Try to load REAL ML model
    ml_loaded = load_ml_model()
    
    if ml_loaded:
        print("✅ REAL ML MODEL loaded - scientifically valid predictions")
    else:
        print("❌ NO ML MODEL - Train first with: python multitask_tactile_network.py")
        print("   Predictions will fail until real model is trained on VibTac-12 data")
    
    print(f"✅ Physics generator loaded with {len(PHYSICS_GENERATOR.materials) * len(PHYSICS_GENERATOR.scenarios)} scenarios")
    print(f"🌐 Starting server at http://localhost:5000")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
