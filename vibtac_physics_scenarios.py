"""
VibTac-12 Physics-Based Scenarios
================================

Comprehensive physics scenarios for all 12 VibTac textures with varied contact conditions.
Each texture has unique material properties and multiple realistic contact scenarios.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class MaterialPhysics:
    """Material physics properties for VibTac-12 textures"""
    stiffness: float        # N/m - Surface stiffness
    damping: float          # Damping coefficient
    friction: float         # Friction coefficient
    roughness: float        # Surface roughness (0-1)
    density: float          # kg/m³
    compressibility: float  # Compression response
    texture_name: str       # VibTac name

@dataclass
class ContactScenario:
    """Individual contact scenario with physics parameters"""
    name: str
    speed_ms: float         # m/s
    mass_kg: float          # kg  
    angle_deg: float        # degrees
    duration_ms: float      # milliseconds
    description: str

class VibTacPhysicsGenerator:
    """Generate realistic physics scenarios for all VibTac-12 textures"""
    
    def __init__(self):
        """Initialize with VibTac-12 material properties from research"""
        self.materials = {
            0: MaterialPhysics(
                stiffness=15000, damping=0.3, friction=0.4, roughness=0.6,
                density=800, compressibility=0.2, texture_name="fabric-1"
            ),
            1: MaterialPhysics(
                stiffness=75000, damping=0.05, friction=0.15, roughness=0.1,
                density=2700, compressibility=0.01, texture_name="aluminum_film"
            ),
            2: MaterialPhysics(
                stiffness=12000, damping=0.35, friction=0.45, roughness=0.7,
                density=750, compressibility=0.25, texture_name="fabric-2"
            ),
            3: MaterialPhysics(
                stiffness=18000, damping=0.25, friction=0.5, roughness=0.5,
                density=850, compressibility=0.18, texture_name="fabric-3"
            ),
            4: MaterialPhysics(
                stiffness=8000, damping=0.45, friction=0.55, roughness=0.8,
                density=600, compressibility=0.4, texture_name="moquette-1"
            ),
            5: MaterialPhysics(
                stiffness=9500, damping=0.4, friction=0.6, roughness=0.75,
                density=650, compressibility=0.35, texture_name="moquette-2"
            ),
            6: MaterialPhysics(
                stiffness=20000, damping=0.2, friction=0.4, roughness=0.45,
                density=900, compressibility=0.15, texture_name="fabric-4"
            ),
            7: MaterialPhysics(
                stiffness=5000, damping=0.6, friction=0.85, roughness=0.9,
                density=400, compressibility=0.6, texture_name="sticky_fabric-5"
            ),
            8: MaterialPhysics(
                stiffness=6000, damping=0.55, friction=0.8, roughness=0.85,
                density=450, compressibility=0.55, texture_name="sticky_fabric"
            ),
            9: MaterialPhysics(
                stiffness=35000, damping=0.15, friction=0.25, roughness=0.3,
                density=1200, compressibility=0.05, texture_name="sparkle_paper-1"
            ),
            10: MaterialPhysics(
                stiffness=32000, damping=0.18, friction=0.3, roughness=0.35,
                density=1150, compressibility=0.08, texture_name="sparkle_paper-2"
            ),
            11: MaterialPhysics(
                stiffness=45000, damping=0.35, friction=0.7, roughness=0.6,
                density=1500, compressibility=0.12, texture_name="toy_tire_rubber"
            )
        }
        
        # Define diverse contact scenarios
        self.scenarios = [
            ContactScenario("gentle_tap", 0.8, 0.005, 15, 50, "Light fingertip tap"),
            ContactScenario("firm_press", 0.3, 0.02, 0, 200, "Firm finger pressure"),
            ContactScenario("quick_poke", 2.5, 0.001, 30, 20, "Quick sharp poke"),
            ContactScenario("sliding_touch", 1.2, 0.01, 45, 150, "Sliding finger motion"),
            ContactScenario("heavy_impact", 5.0, 0.05, 0, 30, "Heavy tool contact"),
            ContactScenario("glancing_strike", 3.5, 0.008, 60, 25, "Angled glancing contact"),
            ContactScenario("sustained_pressure", 0.5, 0.03, 5, 300, "Long sustained contact"),
            ContactScenario("vibrating_touch", 1.8, 0.015, 20, 100, "Vibrating probe contact")
        ]
    
    def get_material(self, texture_id: int) -> MaterialPhysics:
        """Get material properties for texture ID"""
        return self.materials.get(texture_id, self.materials[0])
    
    def get_all_scenarios_for_texture(self, texture_id: int) -> List[Dict]:
        """Generate all physics scenarios for a specific texture"""
        material = self.get_material(texture_id)
        scenarios = []
        
        for scenario in self.scenarios:
            # Calculate physics-based force using realistic spring-damper contact model
            # Normal force from weight and angle
            normal_force = scenario.mass_kg * 9.81 * np.cos(np.radians(scenario.angle_deg))
            
            # Contact compression (spring model: F = k * x)
            compression_depth = 0.001 * material.compressibility  # Convert to meters (0.001-0.5mm)
            spring_force = material.stiffness * compression_depth
            
            # Friction force from sliding
            friction_force = material.friction * normal_force
            
            # Speed-dependent damping force
            damping_force = material.damping * scenario.speed_ms * 10
            
            # Total contact force (clipped to realistic tactile range)
            total_force = np.clip(spring_force + friction_force + damping_force, 0.01, 15.0)
            
            # Generate tactile signal based on material and scenario
            signal = self._generate_material_signal(material, scenario, total_force)
            
            scenarios.append({
                'texture_id': texture_id,
                'texture_name': material.texture_name,
                'scenario_name': scenario.name,
                'scenario_description': scenario.description,
                'physics': {
                    'speed_ms': scenario.speed_ms,
                    'mass_kg': scenario.mass_kg,
                    'angle_deg': scenario.angle_deg,
                    'duration_ms': scenario.duration_ms,
                    'calculated_force_n': total_force,
                    'normal_force_n': float(normal_force),
                    'friction_force_n': float(friction_force),
                    'compression_m': float(compression_depth)
                },
                'material_properties': {
                    'stiffness': material.stiffness,
                    'damping': material.damping,
                    'friction': material.friction,
                    'roughness': material.roughness,
                    'density': material.density,
                    'compressibility': material.compressibility
                },
                'tactile_signal': signal
            })
        
        return scenarios
    
    def get_all_scenarios(self) -> Dict[int, List[Dict]]:
        """Generate all scenarios for all 12 VibTac textures"""
        all_scenarios = {}
        for texture_id in range(12):
            all_scenarios[texture_id] = self.get_all_scenarios_for_texture(texture_id)
        return all_scenarios
    
    def _generate_material_signal(self, material: MaterialPhysics, scenario: ContactScenario, force: float) -> np.ndarray:
        """Generate realistic tactile signal based on material and scenario"""
        sample_rate = 256  # Hz
        duration_s = scenario.duration_ms / 1000.0
        num_samples = int(sample_rate * duration_s)
        
        # Time array
        t = np.linspace(0, duration_s, num_samples)
        
        # Base frequency based on material properties and speed
        speed_factor = max(0.5, min(scenario.speed_ms / 2.0, 3.0))  # 0.5-3x scaling from speed
        base_freq = 20 + (material.stiffness / 1000) * speed_factor  # 20-95+ Hz range
        roughness_freq = 150 + (material.roughness * 200)  # 150-350 Hz range
        
        # Generate multi-component signal with realistic amplitude scaling
        # 1. Impact response (decaying oscillation) - scaled by force (reduced)
        impact_envelope = np.exp(-t * 10)  # Quick decay
        impact_amplitude = 0.5 + (force / 20.0)  # Reduced: 0.5-1.25 amplitude based on force
        impact_signal = impact_amplitude * impact_envelope * np.sin(2 * np.pi * base_freq * t)
        
        # 2. Surface texture response (roughness-dependent) - reduced
        texture_amplitude = material.roughness * 0.4 + 0.2  # Reduced: 0.2-0.6
        texture_signal = texture_amplitude * np.sin(2 * np.pi * roughness_freq * t)
        
        # 3. Material-specific vibrations - reduced
        material_freq = base_freq * (1 + material.compressibility)
        material_signal = 0.3 * np.sin(2 * np.pi * material_freq * t)  # Reduced from 0.5
        
        # 4. Friction-dependent noise - reduced
        friction_noise = (material.friction * 0.1) * np.random.normal(0, 0.1, num_samples)  # Reduced from 0.2
        
        # 5. Damping envelope - material-dependent
        damping_envelope = np.exp(-t * material.damping * 20)
        
        # 6. Speed-based amplitude modulation - reduced
        speed_modulation = 0.6 + (scenario.speed_ms / 5.0)  # Reduced: 0.6-1.0x from speed
        
        # Combine all components
        combined_signal = (impact_signal + texture_signal + material_signal + friction_noise) * damping_envelope * speed_modulation
        
        # Add scenario-specific modulations (reduced amplitudes)
        if scenario.name == "sliding_touch" or "slide" in scenario.name.lower():
            # Add sliding friction modulation with speed dependency
            slide_freq = 5 + scenario.speed_ms * 3
            slide_modulation = 0.2 * np.sin(2 * np.pi * slide_freq * t)  # Reduced from 0.4
            combined_signal += slide_modulation
        elif scenario.name == "vibrating_touch" or "vib" in scenario.name.lower():
            # Add high-frequency vibrations
            vib_freq = 100 + material.stiffness / 500
            vib_signal = 0.3 * np.sin(2 * np.pi * vib_freq * t)  # Reduced from 0.5
            combined_signal += vib_signal
        
        # Apply angle modulation (reduced range)
        angle_factor = np.cos(np.radians(scenario.angle_deg))
        combined_signal *= (0.7 + angle_factor * 0.3)  # Reduced: 0.7-1.0x from angle
        
        # Mass modulation (reduced impact)
        mass_factor = 0.8 + (scenario.mass_kg / 1.0)  # Reduced: normalized around 1.0kg
        combined_signal *= np.clip(mass_factor, 0.5, 1.5)  # Reduced from 3.0 to 1.5
        
        # Ensure signal is finite and within reasonable bounds
        combined_signal = np.nan_to_num(combined_signal)
        combined_signal = np.clip(combined_signal, -10.0, 10.0)
        
        # Pad or trim to exactly 256 samples for model input
        if len(combined_signal) < 256:
            combined_signal = np.pad(combined_signal, (0, 256 - len(combined_signal)), 'constant')
        else:
            combined_signal = combined_signal[:256]
        
        return combined_signal
    
    def generate_dynamic_signal(self, texture_id: int, speed_ms: float, mass_kg: float, angle_deg: float, scenario_name: str = "dynamic") -> Dict:
        """Generate dynamic tactile signal using real-time user parameters"""
        material = self.get_material(texture_id)
        
        # Create dynamic scenario from user parameters
        duration_ms = 150.0  # Fixed duration for consistency
        dynamic_scenario = ContactScenario(
            name=scenario_name,
            speed_ms=speed_ms,
            mass_kg=mass_kg, 
            angle_deg=angle_deg,
            duration_ms=duration_ms,
            description=f"Dynamic {scenario_name} at {speed_ms}m/s"
        )
        
        # Calculate physics-based force using realistic spring-damper contact model
        # For tactile interactions (not impacts), use contact force from normal load + friction
        
        # Normal force from weight and angle
        normal_force = mass_kg * 9.81 * np.cos(np.radians(angle_deg))  # Weight component perpendicular to surface
        
        # Contact compression (spring model: F = k * x)
        # Typical tactile compression is 0.5-2mm
        compression_depth = 0.001 * material.compressibility  # Convert to meters (0.001-0.5mm range)
        spring_force = material.stiffness * compression_depth  # N = (N/m) * m
        
        # Friction force from sliding
        friction_force = material.friction * normal_force
        
        # Speed-dependent damping force (B * v)
        damping_force = material.damping * speed_ms * 10  # Scaled damping
        
        # Total contact force (combine spring, friction, damping)
        # For tactile sensing, typical forces are 0.1-10N
        total_force = np.clip(spring_force + friction_force + damping_force, 0.01, 15.0)
        
        # Generate tactile signal based on material and dynamic scenario
        signal = self._generate_material_signal(material, dynamic_scenario, total_force)
        
        return {
            'texture_id': texture_id,
            'texture_name': material.texture_name,
            'scenario_name': dynamic_scenario.name,
            'tactile_signal': signal.tolist(),
            'physics': {
                'calculated_force_n': float(total_force),
                'speed_ms': speed_ms,
                'mass_kg': mass_kg,
                'angle_deg': angle_deg,
                'normal_force_n': float(normal_force),
                'friction_force_n': float(friction_force),
                'compression_m': float(compression_depth)
            },
            'material_properties': {
                'stiffness': material.stiffness,
                'damping': material.damping,
                'friction': material.friction,
                'roughness': material.roughness
            }
        }
    
    def generate_scenario_summary(self) -> Dict:
        """Generate summary of all scenarios for analysis"""
        summary = {
            'total_textures': 12,
            'scenarios_per_texture': len(self.scenarios),
            'total_scenarios': 12 * len(self.scenarios),
            'force_range': {
                'min_force': float('inf'),
                'max_force': 0.0,
                'avg_force': 0.0
            },
            'speed_range': {
                'min_speed': min(s.speed_ms for s in self.scenarios),
                'max_speed': max(s.speed_ms for s in self.scenarios),
            },
            'mass_range': {
                'min_mass': min(s.mass_kg for s in self.scenarios),
                'max_mass': max(s.mass_kg for s in self.scenarios),
            },
            'texture_properties': {}
        }
        
        # Calculate force statistics across all scenarios
        all_forces = []
        for texture_id in range(12):
            scenarios = self.get_all_scenarios_for_texture(texture_id)
            material = self.get_material(texture_id)
            
            texture_forces = [s['physics']['calculated_force_n'] for s in scenarios]
            all_forces.extend(texture_forces)
            
            summary['texture_properties'][texture_id] = {
                'name': material.texture_name,
                'force_range': [min(texture_forces), max(texture_forces)],
                'avg_force': np.mean(texture_forces),
                'stiffness': material.stiffness,
                'friction': material.friction,
                'roughness': material.roughness
            }
        
        summary['force_range'] = {
            'min_force': min(all_forces),
            'max_force': max(all_forces),
            'avg_force': np.mean(all_forces)
        }
        
        return summary

if __name__ == "__main__":
    # Demo: Generate scenarios for all textures
    generator = VibTacPhysicsGenerator()
    
    print("🔬 VibTac-12 Physics Scenario Generator")
    print("=" * 50)
    
    # Generate summary
    summary = generator.generate_scenario_summary()
    print(f"\n📊 Summary:")
    print(f"   • Total scenarios: {summary['total_scenarios']}")
    print(f"   • Force range: {summary['force_range']['min_force']:.2f} - {summary['force_range']['max_force']:.2f} N")
    print(f"   • Speed range: {summary['speed_range']['min_speed']} - {summary['speed_range']['max_speed']} m/s")
    print(f"   • Mass range: {summary['mass_range']['min_mass']} - {summary['mass_range']['max_mass']} kg")
    
    # Show sample scenarios for first texture
    print(f"\n🧵 Sample scenarios for texture 0 (fabric-1):")
    scenarios = generator.get_all_scenarios_for_texture(0)
    
    for i, scenario in enumerate(scenarios[:3]):  # Show first 3
        print(f"   {i+1}. {scenario['scenario_name']}: {scenario['scenario_description']}")
        print(f"      Speed: {scenario['physics']['speed_ms']} m/s, Force: {scenario['physics']['calculated_force_n']:.2f} N")
        print(f"      Signal shape: {scenario['tactile_signal'].shape}")
    
    print(f"\n✅ Ready to generate realistic tactile data for all 12 VibTac textures!")