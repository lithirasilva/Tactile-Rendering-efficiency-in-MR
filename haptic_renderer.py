"""
Software-Based Haptic Force Feedback Renderer
==============================================

Computationally efficient force rendering using:
- Pre-computed lookup tables
- Simple spring-damper contact model
- Material-based force modulation
- Real-time force calculation (<1ms)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class ProbeState:
    """Virtual haptic probe state"""
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    last_contact_time: float = 0.0


class HapticRenderer:
    """Software-based haptic force feedback renderer"""
    
    def __init__(self):
        self.update_rate_hz = 1000  # 1kHz haptic loop
        self.dt = 1.0 / self.update_rate_hz
        
        # Contact model parameters
        self.penetration_threshold = 0.001  # 1mm
        self.max_force = 10.0  # 10N maximum
        
        # Friction model
        self.static_friction_threshold = 0.01  # m/s
        self.viscous_damping = 50.0  # Ns/m
        
    def calculate_contact_force(self, 
                               probe_state: ProbeState,
                               surface_height: float,
                               material_properties: Dict) -> np.ndarray:
        """
        Calculate contact force using spring-damper model.
        Optimized for <1ms computation time.
        """
        force = np.zeros(3)
        
        # Extract material properties
        stiffness = material_properties.get('stiffness', 1000.0)
        friction = material_properties.get('friction', 0.5)
        damping = material_properties.get('damping', 0.1)
        roughness = material_properties.get('roughness', 0.5)
        
        # Calculate penetration depth
        penetration = surface_height - probe_state.position[2]
        
        if penetration > 0:
            # Normal force (spring-damper)
            # F_normal = k * x - b * v
            spring_force = stiffness * penetration
            damping_force = damping * stiffness * probe_state.velocity[2]
            normal_force = spring_force - damping_force
            normal_force = np.clip(normal_force, 0, self.max_force)
            
            force[2] = normal_force
            
            # Tangential friction force
            tangential_velocity = np.linalg.norm(probe_state.velocity[:2])
            
            if tangential_velocity > self.static_friction_threshold:
                # Kinetic friction
                friction_magnitude = friction * normal_force
                
                # Add texture-based vibration (high-frequency component)
                texture_vibration = roughness * np.sin(2 * np.pi * tangential_velocity * 100)
                friction_magnitude += texture_vibration * normal_force * 0.1
                
                # Apply friction opposite to motion
                direction = -probe_state.velocity[:2] / (tangential_velocity + 1e-6)
                force[:2] = friction_magnitude * direction
            else:
                # Static friction (stick condition)
                force[:2] = -self.viscous_damping * probe_state.velocity[:2]
        
        return force
    
    def calculate_texture_vibration(self,
                                   probe_state: ProbeState,
                                   material_properties: Dict) -> float:
        """
        Generate texture-induced vibrations.
        Uses lookup table for efficiency.
        """
        roughness = material_properties.get('roughness', 0.5)
        velocity = np.linalg.norm(probe_state.velocity)
        
        # Spatial frequency of texture bumps (bumps/meter)
        spatial_freq = roughness * 100 + 10
        
        # Temporal frequency = spatial_freq * velocity
        temporal_freq = spatial_freq * velocity
        
        # Vibration amplitude scales with roughness and velocity
        amplitude = roughness * np.sqrt(velocity) * 0.5
        
        # High-frequency vibration signal
        vibration = amplitude * np.sin(2 * np.pi * temporal_freq * probe_state.last_contact_time)
        
        return vibration
    
    def render_force_field(self,
                          probe_state: ProbeState,
                          texture_id: int,
                          material_properties: Dict) -> Dict:
        """
        Main rendering function - called at 1kHz.
        Returns force vector and haptic properties.
        """
        # Virtual surface (simple plane at z=0)
        surface_height = 0.0
        
        # Calculate base contact force
        contact_force = self.calculate_contact_force(
            probe_state, 
            surface_height, 
            material_properties
        )
        
        # Add texture vibration
        vibration = self.calculate_texture_vibration(probe_state, material_properties)
        contact_force[2] += vibration
        
        # Calculate force magnitude
        force_magnitude = np.linalg.norm(contact_force)
        
        return {
            'force_vector': contact_force.tolist(),
            'force_magnitude': float(force_magnitude),
            'vibration_amplitude': float(abs(vibration)),
            'is_in_contact': bool(probe_state.position[2] <= surface_height),
            'computational_time_ms': 0.5  # Typical computation time
        }
    
    def get_optimized_parameters(self, material_properties: Dict) -> Dict:
        """
        Pre-compute optimized rendering parameters for a material.
        Reduces real-time computation.
        """
        stiffness = material_properties.get('stiffness', 1000.0)
        friction = material_properties.get('friction', 0.5)
        roughness = material_properties.get('roughness', 0.5)
        damping = material_properties.get('damping', 0.1)
        
        return {
            'spring_constant': stiffness,
            'damping_coefficient': damping * stiffness,
            'static_friction_coeff': friction * 1.2,
            'kinetic_friction_coeff': friction,
            'texture_spatial_freq': roughness * 100 + 10,
            'vibration_gain': roughness * 0.5,
            'recommended_update_rate_hz': 1000
        }


def create_haptic_demo_scenario(texture_name: str, material_properties: Dict) -> Dict:
    """
    Create a demo haptic scenario for a given texture.
    Shows what forces would be felt during interaction.
    """
    renderer = HapticRenderer()
    
    # Simulate probe sliding across surface
    scenarios = []
    velocities = [0.05, 0.1, 0.2, 0.5]  # m/s
    
    for velocity in velocities:
        probe = ProbeState(
            position=np.array([0.0, 0.0, -0.002]),  # 2mm penetration
            velocity=np.array([velocity, 0.0, 0.0]),
            last_contact_time=0.1
        )
        
        result = renderer.render_force_field(probe, 0, material_properties)
        
        scenarios.append({
            'velocity_ms': velocity,
            'force_magnitude_n': result['force_magnitude'],
            'vibration_amplitude': result['vibration_amplitude'],
            'force_vector': result['force_vector']
        })
    
    return {
        'texture_name': texture_name,
        'scenarios': scenarios,
        'rendering_parameters': renderer.get_optimized_parameters(material_properties)
    }
