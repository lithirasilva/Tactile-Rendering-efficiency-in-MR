"""
Multi-Task Neural Network for Tactile Perception
================================================

Proper Architecture:
Input: Raw X,Y,Z acceleration signals (256 samples each)
Output: Texture classification + Force regression

Real Use Case: 
"An object touches the sensor - what material is it and how hard did it touch?"

No circular logic. No pre-selected textures. Just raw sensor → predictions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TactileDataset(Dataset):
    """Dataset for raw tactile signals"""
    
    def __init__(self, signals, textures, forces):
        """
        Args:
            signals: (N, 768) - 256 samples each for X, Y, Z
            textures: (N,) - texture labels 0-11
            forces: (N,) - force values in Newtons
        """
        self.signals = torch.FloatTensor(signals)
        self.textures = torch.LongTensor(textures)  
        self.forces = torch.FloatTensor(forces)
        
    def __len__(self):
        return len(self.signals)
        
    def __getitem__(self, idx):
        return self.signals[idx], self.textures[idx], self.forces[idx]


class SharedFeatureExtractor(nn.Module):
    """Shared CNN feature extractor for tactile signals"""
    
    def __init__(self, input_size=768, hidden_size=128, feature_size=42):
        super().__init__()
        
        # 1D CNN for temporal patterns in XYZ acceleration
        self.conv1 = nn.Conv1d(3, 16, kernel_size=5, padding=2)  # 3 channels: X,Y,Z
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(32)  # Adaptive pooling
        self.dropout = nn.Dropout(0.3)
        
        # Dense layers for feature extraction
        self.fc1 = nn.Linear(64 * 32, hidden_size)
        self.fc2 = nn.Linear(hidden_size, feature_size)
        
    def forward(self, x):
        # Reshape input: (batch, 768) -> (batch, 3, 256) 
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 256)  # 3 channels (X,Y,Z), 256 time steps each
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global pooling
        x = self.pool(x)  # (batch, 64, 32)
        x = x.view(batch_size, -1)  # Flatten
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        features = self.fc2(x)
        
        return features


class TextureHead(nn.Module):
    """Texture classification head"""
    
    def __init__(self, feature_size=42, num_textures=12):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_textures)
        )
        
    def forward(self, features):
        return self.classifier(features)


class ForceHead(nn.Module):
    """Force regression head"""
    
    def __init__(self, feature_size=42):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, features):
        return self.regressor(features).squeeze(-1)


class MultiTaskTactileNet(nn.Module):
    """Complete multi-task network"""
    
    def __init__(self, feature_size=42, num_textures=12):
        super().__init__()
        self.feature_extractor = SharedFeatureExtractor(feature_size=feature_size)
        self.texture_head = TextureHead(feature_size, num_textures)
        self.force_head = ForceHead(feature_size)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        texture_logits = self.texture_head(features)
        force_pred = self.force_head(features)
        return texture_logits, force_pred, features


class MultiTaskTrainer:
    """Trainer for the multi-task tactile network"""
    
    def __init__(self, learning_rate=0.001, texture_weight=1.0, force_weight=1.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.texture_weight = texture_weight
        self.force_weight = force_weight
        
        # Model
        self.model = MultiTaskTactileNet().to(self.device)
        
        # Loss functions
        self.texture_loss_fn = nn.CrossEntropyLoss()
        self.force_loss_fn = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Multi-task model initialized on {self.device}")
        self._log_model_size()
        
    def _log_model_size(self):
        """Log computational requirements"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate memory usage (MB)
        param_memory = total_params * 4 / 1024 / 1024  # 4 bytes per float32
        
        logger.info(f"💻 COMPUTE RESOURCES:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Model memory: {param_memory:.2f} MB")
        logger.info(f"   Device: {self.device}")
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        texture_correct = 0
        total_samples = 0
        
        for signals, textures, forces in train_loader:
            signals = signals.to(self.device)
            textures = textures.to(self.device)
            forces = forces.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            texture_logits, force_pred, features = self.model(signals)
            
            # Multi-task loss
            texture_loss = self.texture_loss_fn(texture_logits, textures)
            force_loss = self.force_loss_fn(force_pred, forces)
            total_loss_batch = (
                self.texture_weight * texture_loss + 
                self.force_weight * force_loss
            )
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += total_loss_batch.item()
            texture_pred = torch.argmax(texture_logits, dim=1)
            texture_correct += (texture_pred == textures).sum().item()
            total_samples += textures.size(0)
            
        avg_loss = total_loss / len(train_loader)
        texture_acc = texture_correct / total_samples
        
        return avg_loss, texture_acc
        
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        texture_correct = 0
        total_samples = 0
        force_preds = []
        force_targets = []
        
        with torch.no_grad():
            for signals, textures, forces in val_loader:
                signals = signals.to(self.device)
                textures = textures.to(self.device) 
                forces = forces.to(self.device)
                
                texture_logits, force_pred, features = self.model(signals)
                
                texture_loss = self.texture_loss_fn(texture_logits, textures)
                force_loss = self.force_loss_fn(force_pred, forces)
                total_loss += (
                    self.texture_weight * texture_loss + 
                    self.force_weight * force_loss
                ).item()
                
                texture_pred = torch.argmax(texture_logits, dim=1)
                texture_correct += (texture_pred == textures).sum().item()
                total_samples += textures.size(0)
                
                force_preds.extend(force_pred.cpu().numpy())
                force_targets.extend(forces.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        texture_acc = texture_correct / total_samples
        force_r2 = r2_score(force_targets, force_preds)
        force_mae = mean_absolute_error(force_targets, force_preds)
        
        return avg_loss, texture_acc, force_r2, force_mae
        
    def measure_inference_speed(self, test_loader):
        """Measure inference speed and compute usage"""
        self.model.eval()
        times = []
        
        with torch.no_grad():
            for signals, _, _ in test_loader:
                signals = signals.to(self.device)
                
                # Measure single sample inference
                single_sample = signals[:1]  # Just first sample
                
                start_time = time.perf_counter()
                texture_logits, force_pred, features = self.model(single_sample)
                inference_time = (time.perf_counter() - start_time) * 1000  # ms
                
                times.append(inference_time)
                
                if len(times) >= 100:  # Test 100 samples
                    break
                    
        avg_inference = np.mean(times)
        std_inference = np.std(times)
        
        logger.info(f"⚡ INFERENCE PERFORMANCE:")
        logger.info(f"   Average inference: {avg_inference:.3f} ± {std_inference:.3f} ms")
        logger.info(f"   Throughput: ~{1000/avg_inference:.1f} samples/second")
        
        return avg_inference
        

def load_real_vibtac_data():
    """
    Load REAL VibTac-12 dataset - NO SYNTHETIC DATA
    
    Returns actual sensor measurements from research dataset
    """
    from real_vibtac_loader import load_real_vibtac_dataset
    
    logger.info("🔄 Loading REAL VibTac-12 dataset...")
    
    try:
        signals, textures, forces = load_real_vibtac_dataset(".", source="auto")
        
        if signals is None:
            raise ValueError("Failed to load real VibTac dataset")
        
        logger.info(f"✅ Real VibTac dataset loaded:")
        logger.info(f"   Samples: {len(signals)}")
        logger.info(f"   Textures: {len(np.unique(textures))} classes")
        logger.info(f"   Force range: {forces.min():.2f} - {forces.max():.2f} N")
        
        # Ensure signals are properly formatted for neural network
        # Pad/truncate to standard size if needed
        target_size = 768  # 3 * 256 samples per axis
        processed_signals = []
        
        for signal in signals:
            if len(signal) < target_size:
                # Pad with zeros if too short
                padded = np.zeros(target_size)
                padded[:len(signal)] = signal
                processed_signals.append(padded)
            elif len(signal) > target_size:
                # Truncate if too long
                processed_signals.append(signal[:target_size])
            else:
                processed_signals.append(signal)
        
        signals = np.array(processed_signals)
        
        logger.info(f"📊 Processed dataset:")
        logger.info(f"   Signal shape: {signals[0].shape}")
        logger.info(f"   All signals normalized to {target_size} samples")
        
        return signals, textures, forces
        
    except Exception as e:
        logger.error(f"❌ Failed to load real VibTac dataset: {e}")
        logger.error("Ensure VibTac-12 dataset is available in workspace")
        raise


def main():
    """Main training pipeline - REAL DATA ONLY"""
    logger.info("="*70)
    logger.info("MULTI-TASK TACTILE NEURAL NETWORK - REAL DATA TRAINING")
    logger.info("="*70)
    logger.info("Use Case: Real VibTac-12 sensor data → Predict material + force")
    logger.info("Input: Actual X,Y,Z acceleration (from research dataset)")
    logger.info("Output: Texture class (0-11) + Force value (Newtons)")
    logger.info("="*70)
    
    # Load REAL VibTac dataset - NO SYNTHETIC DATA
    logger.info("🔄 Loading REAL VibTac-12 dataset...")
    signals, textures, forces = load_real_vibtac_data()
    
    logger.info(f"📊 REAL Dataset: {len(signals)} samples")
    logger.info(f"   Signal shape: {signals[0].shape} (768 = 3 × 256)")
    logger.info(f"   Textures: 0-{max(textures)} ({len(np.unique(textures))} classes)")
    logger.info(f"   Forces: {forces.min():.1f} - {forces.max():.1f} N")
    logger.info("   ✅ NO SYNTHETIC DATA - All measurements from real sensors")
    
    # Split data
    X_train, X_test, y_tex_train, y_tex_test, y_force_train, y_force_test = train_test_split(
        signals, textures, forces, test_size=0.2, random_state=42, stratify=textures
    )
    
    # Create data loaders
    train_dataset = TactileDataset(X_train, y_tex_train, y_force_train)
    test_dataset = TactileDataset(X_test, y_tex_test, y_force_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize trainer
    trainer = MultiTaskTrainer(learning_rate=0.001, texture_weight=1.0, force_weight=0.1)
    
    # Training
    logger.info(f"\n🚀 Training on REAL VibTac-12 data...")
    epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss, train_texture_acc = trainer.train_epoch(train_loader)
        val_loss, val_texture_acc, val_force_r2, val_force_mae = trainer.validate(test_loader)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, "
                       f"Val Texture Acc={val_texture_acc:.3f}, Val Force R²={val_force_r2:.3f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(trainer.model.state_dict(), 'multitask_tactile_model_REAL.pth')
    
    # Final evaluation
    logger.info(f"\n📈 FINAL RESULTS (Real VibTac-12 Data):")
    final_loss, final_texture_acc, final_force_r2, final_force_mae = trainer.validate(test_loader)
    logger.info(f"   Texture Accuracy: {final_texture_acc:.1%}")
    logger.info(f"   Force R²: {final_force_r2:.3f}")
    logger.info(f"   Force MAE: {final_force_mae:.2f} N")
    
    # Measure inference speed
    inference_time = trainer.measure_inference_speed(test_loader)
    
    logger.info(f"\n💾 Model saved: multitask_tactile_model_REAL.pth")
    logger.info(f"🎯 Trained on REAL VibTac-12 data - scientifically valid!")
    
    # Save training metadata
    metadata = {
        "dataset": "VibTac-12 Real",
        "samples_total": len(signals),
        "samples_train": len(X_train),
        "samples_test": len(X_test),
        "texture_accuracy": float(final_texture_acc),
        "force_r2": float(final_force_r2),
        "force_mae": float(final_force_mae),
        "inference_time_ms": float(inference_time),
        "training_date": pd.Timestamp.now().isoformat()
    }
    
    import json
    with open('real_vibtac_training_results.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"📊 Training metadata saved: real_vibtac_training_results.json")
    

if __name__ == "__main__":
    main()