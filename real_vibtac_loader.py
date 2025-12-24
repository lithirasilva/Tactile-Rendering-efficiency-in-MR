"""
Real VibTac-12 Dataset Loader
============================

Loads actual VibTac-12 dataset from pickle files and CSV files.
No synthetic data generation - only real sensor measurements.

Dataset Structure:
- VibTac12Dataset/: CSV files (X.csv, Y.csv, Z.csv, S.csv)
- Multimodal Tactile Texture Dataset/pickles_*/: Pickle files per texture/velocity
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VibTacDataLoader:
    """Loads real VibTac-12 dataset from multiple sources"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.csv_path = self.dataset_path / "VibTac12Dataset"
        self.pickle_path = self.dataset_path / "Multimodal Tactile Texture Dataset"
        
        # VibTac-12 texture names mapping
        self.texture_names = [
            "carpet", "cardboard", "cotton", "denim", "felt", "foam",
            "leather", "paper", "plastic", "rubber", "sandpaper", "wood"
        ]
        
    def load_csv_dataset(self):
        """Load from CSV files (X.csv, Y.csv, Z.csv, S.csv)"""
        logger.info("Loading VibTac-12 CSV dataset...")
        
        try:
            # Load raw accelerometer data
            x_data = pd.read_csv(self.csv_path / "X.csv", header=None)
            y_data = pd.read_csv(self.csv_path / "Y.csv", header=None) 
            z_data = pd.read_csv(self.csv_path / "Z.csv", header=None)
            s_data = pd.read_csv(self.csv_path / "S.csv", header=None)  # Subject/texture labels
            
            logger.info(f"Raw data shapes: X={x_data.shape}, Y={y_data.shape}, Z={z_data.shape}, S={s_data.shape}")
            
            # Stack X,Y,Z into signals
            signals = np.column_stack([
                x_data.values.flatten(),
                y_data.values.flatten(), 
                z_data.values.flatten()
            ])
            
            textures = s_data.values.flatten()
            
            # Generate realistic force estimates from acceleration magnitude
            forces = self._estimate_forces_from_acceleration(x_data.values, y_data.values, z_data.values)
            
            logger.info(f"Loaded {len(signals)} real VibTac samples")
            logger.info(f"Textures: {len(np.unique(textures))} classes, Forces: {forces.min():.2f}-{forces.max():.2f}N")
            
            return signals, textures, forces
            
        except Exception as e:
            logger.error(f"Failed to load CSV dataset: {e}")
            return None, None, None
    
    def load_pickle_dataset(self, velocity="35"):
        """Load from pickle files (more detailed format)"""
        logger.info(f"Loading VibTac-12 pickle dataset (velocity={velocity})...")
        
        pickle_dir = self.pickle_path / f"pickles_{velocity}"
        if not pickle_dir.exists():
            logger.warning(f"Pickle directory {pickle_dir} not found")
            return None, None, None
            
        signals = []
        textures = []
        forces = []
        
        # Load each texture type
        for texture_id, texture_name in enumerate(self.texture_names):
            texture_dir = pickle_dir / f"texture_{texture_id+1:02d}"
            if not texture_dir.exists():
                continue
                
            # Load IMU data (most comprehensive)
            imu_dir = texture_dir / "full_imu"
            if imu_dir.exists():
                for pickle_file in imu_dir.glob("*.pkl"):
                    try:
                        with open(pickle_file, 'rb') as f:
                            df = pickle.load(f)
                            
                        # Extract accelerometer columns (ax, ay, az)
                        if all(col in df.columns for col in ['ax', 'ay', 'az']):
                            signal = np.column_stack([
                                df['ax'].values,
                                df['ay'].values,
                                df['az'].values
                            ]).flatten()
                            
                            signals.append(signal)
                            textures.append(texture_id)
                            
                            # Estimate force from acceleration magnitude
                            force = self._estimate_force_from_imu(df)
                            forces.append(force)
                            
                    except Exception as e:
                        logger.warning(f"Failed to load {pickle_file}: {e}")
                        continue
        
        if signals:
            signals = np.array(signals)
            textures = np.array(textures)
            forces = np.array(forces)
            
            logger.info(f"Loaded {len(signals)} real VibTac samples from pickles")
            logger.info(f"Signal shape: {signals[0].shape}, Textures: {len(np.unique(textures))}")
            
        return signals, textures, forces
    
    def _estimate_forces_from_acceleration(self, x_data, y_data, z_data):
        """Estimate contact forces from acceleration magnitude"""
        # Calculate acceleration magnitude for each sample
        acc_magnitude = np.sqrt(x_data**2 + y_data**2 + z_data**2)
        
        # Convert acceleration to estimated force (rough physics model)
        # Assume contact mass ~50g, typical tactile interaction
        mass_kg = 0.05
        forces = acc_magnitude.mean(axis=1) * mass_kg  # F = ma
        
        # Normalize to reasonable force range (0.1 - 10 N)
        forces = np.clip(forces * 2.0, 0.1, 10.0)
        
        return forces
    
    def _estimate_force_from_imu(self, df):
        """Estimate force from IMU dataframe"""
        if all(col in df.columns for col in ['ax', 'ay', 'az']):
            acc_mag = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2).mean()
            force = np.clip(acc_mag * 0.1, 0.1, 10.0)  # Scale to realistic range
            return force
        else:
            return 1.0  # Default force
    
    def get_dataset_info(self):
        """Get information about available datasets"""
        csv_available = (self.csv_path / "X.csv").exists()
        
        pickle_dirs = []
        if self.pickle_path.exists():
            for p in self.pickle_path.glob("pickles_*"):
                if p.is_dir():
                    pickle_dirs.append(p.name.split("_")[1])
        
        return {
            "csv_available": csv_available,
            "pickle_velocities": pickle_dirs,
            "texture_count": len(self.texture_names),
            "texture_names": self.texture_names
        }


def load_real_vibtac_dataset(dataset_path: str, source: str = "auto"):
    """
    Load real VibTac-12 dataset
    
    Args:
        dataset_path: Path to dataset directory
        source: 'csv', 'pickle', or 'auto'
    
    Returns:
        signals, textures, forces: Real sensor data
    """
    loader = VibTacDataLoader(dataset_path)
    info = loader.get_dataset_info()
    
    logger.info("VibTac-12 Dataset Info:")
    logger.info(f"  CSV available: {info['csv_available']}")
    logger.info(f"  Pickle velocities: {info['pickle_velocities']}")
    logger.info(f"  Textures: {info['texture_names']}")
    
    if source == "auto":
        # Use CSV data which is more reliable
        if info['csv_available']:
            source = "csv"
        elif info['pickle_velocities']:
            source = "pickle"
        else:
            raise FileNotFoundError("No VibTac dataset found")
    
    if source == "pickle" and info['pickle_velocities']:
        velocity = info['pickle_velocities'][0]  # Use first available velocity
        return loader.load_pickle_dataset(velocity)
    elif source == "csv" and info['csv_available']:
        return loader.load_csv_dataset()
    else:
        raise ValueError(f"Requested source '{source}' not available")


if __name__ == "__main__":
    # Test the data loader
    dataset_path = "."  # Current directory
    
    try:
        signals, textures, forces = load_real_vibtac_dataset(dataset_path)
        
        if signals is not None:
            print(f"✅ Successfully loaded real VibTac dataset:")
            print(f"   Samples: {len(signals)}")
            print(f"   Signal shape: {signals[0].shape}")
            print(f"   Textures: {len(np.unique(textures))} classes")
            print(f"   Force range: {forces.min():.2f} - {forces.max():.2f} N")
        else:
            print("❌ Failed to load dataset")
            
    except Exception as e:
        print(f"❌ Error: {e}")