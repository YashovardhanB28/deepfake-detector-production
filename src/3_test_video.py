"""
INTERACTIVE DEEPFAKE DETECTOR TESTER
Detailed analysis with visual explanations

Features:
- Frame-by-frame predictions
- Grad-CAM visualization (shows what model looks at)
- Confidence scores and reasoning
- Temporal consistency analysis
- Final verdict with evidence

Author: Yashovardhan Bangur
Version: V1_INTERACTIVE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

class DeepfakeDetector(nn.Module):
    """Same architecture as training"""
    def __init__(self, num_classes=2, dropout=0.3):
        super(DeepfakeDetector, self).__init__()
        
        self.backbone = models.efficientnet_b1(weights=None)
        self._disable_inplace_ops(self.backbone)
        
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(512, num_classes)
        )
    
    def _disable_inplace_ops(self, module):
        for child_name, child in module.named_children():
            if isinstance(child, (nn.ReLU, nn.ReLU6, nn.SiLU)):
                setattr(module, child_name, type(child)(inplace=False))
            else:
                self._disable_inplace_ops(child)
    
    def forward(self, x):
        return self.backbone(x)

class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks (use full backward hook to avoid warnings)
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class):
        """Generate CAM for target class"""
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Get weights
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[1, 2])  # [C]
        
        # Weighted combination of activation maps (ensure same device)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

class DeepfakeAnalyzer:
    """Main analyzer class"""
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print("📦 Loading model...")
        self.model = DeepfakeDetector(num_classes=2)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"   ✓ Model loaded (epoch {checkpoint['epoch']}, val_acc: {100*checkpoint['metrics']['accuracy']:.2f}%)")
        
        # Setup Grad-CAM (target last conv layer before classifier)
        target_layer = self.model.backbone.features[-1]
        self.grad_cam = GradCAM(self.model, target_layer)
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['REAL', 'FAKE']
    
    def extract_frames(self, video_path, num_frames=10):
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"\n📹 Video Info:")
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Duration: {duration:.2f}s")
        
        # Get frame indices
        if total_frames < num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        frame_times = []
        
        print(f"\n🎬 Extracting {len(indices)} frames...")
        for idx in tqdm(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_times.append(idx / fps if fps > 0 else 0)
        
        cap.release()
        return frames, frame_times
    
    def analyze_frame(self, frame, return_cam=True):
        """Analyze single frame with optional Grad-CAM"""
        # Prepare image
        img_pil = Image.fromarray(frame)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        # Prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        prediction = {
            'class': self.class_names[predicted.item()],
            'class_id': predicted.item(),
            'confidence': confidence.item(),
            'probabilities': {
                'REAL': probabilities[0, 0].item(),
                'FAKE': probabilities[0, 1].item()
            }
        }
        
        # Generate Grad-CAM if requested
        cam = None
        if return_cam:
            cam = self.grad_cam.generate_cam(img_tensor, predicted.item())
        
        return prediction, cam
    
    def create_heatmap_overlay(self, frame, cam, alpha=0.5):
        """Create heatmap overlay on frame"""
        # Resize CAM to match frame
        cam_resized = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        return overlay, heatmap
    
    def get_reasoning(self, prediction, temporal_stats=None):
        """Generate human-readable reasoning"""
        reasons = []
        
        conf = prediction['confidence']
        pred_class = prediction['class']
        
        # Confidence-based reasoning
        if conf > 0.95:
            reasons.append(f"🔴 VERY HIGH confidence ({conf*100:.1f}%) - clear {pred_class.lower()} characteristics")
        elif conf > 0.85:
            reasons.append(f"🟠 HIGH confidence ({conf*100:.1f}%) - strong {pred_class.lower()} indicators")
        elif conf > 0.70:
            reasons.append(f"🟡 MODERATE confidence ({conf*100:.1f}%) - noticeable {pred_class.lower()} patterns")
        else:
            reasons.append(f"🟢 LOW confidence ({conf*100:.1f}%) - weak/ambiguous signals")
        
        # Probability analysis
        fake_prob = prediction['probabilities']['FAKE']
        real_prob = prediction['probabilities']['REAL']
        
        if abs(fake_prob - real_prob) < 0.2:
            reasons.append("⚠️  Close call - probabilities are similar")
        
        # Temporal consistency (if available)
        if temporal_stats:
            consistency = temporal_stats['prediction_consistency']
            if consistency > 0.8:
                reasons.append(f"✅ Highly consistent across frames ({consistency*100:.1f}%)")
            elif consistency < 0.6:
                reasons.append(f"⚠️  Inconsistent predictions across frames ({consistency*100:.1f}%)")
        
        return reasons
    
    def analyze_video(self, video_path, num_frames=10, save_visualizations=True):
        """Complete video analysis"""
        print("\n" + "="*80)
        print(f"🎯 ANALYZING: {Path(video_path).name}")
        print("="*80)
        
        # Extract frames
        frames, frame_times = self.extract_frames(video_path, num_frames)
        
        if not frames:
            print("❌ No frames extracted!")
            return None
        
        # Analyze each frame
        print("\n🧠 Analyzing frames...")
        results = []
        
        for i, (frame, time) in enumerate(zip(frames, frame_times)):
            prediction, cam = self.analyze_frame(frame, return_cam=True)
            
            results.append({
                'frame_id': i,
                'time': time,
                'frame': frame,
                'prediction': prediction,
                'cam': cam
            })
        
        # Temporal analysis
        predictions_list = [r['prediction']['class_id'] for r in results]
        confidences = [r['prediction']['confidence'] for r in results]
        
        # Count predictions
        fake_count = sum(1 for p in predictions_list if p == 1)
        real_count = sum(1 for p in predictions_list if p == 0)
        
        # Consistency
        most_common = max(fake_count, real_count)
        consistency = most_common / len(predictions_list)
        
        temporal_stats = {
            'fake_frames': fake_count,
            'real_frames': real_count,
            'prediction_consistency': consistency,
            'avg_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
        
        # Final verdict
        final_prediction = 'FAKE' if fake_count > real_count else 'REAL'
        final_confidence = temporal_stats['avg_confidence']
        
        # Print results
        self.print_results(results, temporal_stats, final_prediction, final_confidence)
        
        # Visualizations
        if save_visualizations:
            output_dir = Path('analysis_results') / Path(video_path).stem
            output_dir.mkdir(parents=True, exist_ok=True)
            self.save_visualizations(results, temporal_stats, output_dir)
        
        return {
            'results': results,
            'temporal_stats': temporal_stats,
            'final_prediction': final_prediction,
            'final_confidence': final_confidence
        }
    
    def print_results(self, results, temporal_stats, final_prediction, final_confidence):
        """Print detailed results"""
        print("\n" + "="*80)
        print("📊 FRAME-BY-FRAME ANALYSIS")
        print("="*80)
        
        for r in results:
            pred = r['prediction']
            time = r['time']
            
            status = "🔴" if pred['class'] == 'FAKE' else "🟢"
            print(f"\nFrame {r['frame_id']+1} @ {time:.2f}s:")
            print(f"   {status} Prediction: {pred['class']}")
            print(f"   Confidence: {pred['confidence']*100:.2f}%")
            print(f"   Probabilities: Real={pred['probabilities']['REAL']*100:.1f}%, Fake={pred['probabilities']['FAKE']*100:.1f}%")
            
            # Reasoning
            reasons = self.get_reasoning(pred, temporal_stats)
            for reason in reasons:
                print(f"   • {reason}")
        
        print("\n" + "="*80)
        print("🎯 TEMPORAL CONSISTENCY ANALYSIS")
        print("="*80)
        print(f"   Total frames analyzed: {len(results)}")
        print(f"   Fake detections: {temporal_stats['fake_frames']} ({100*temporal_stats['fake_frames']/len(results):.1f}%)")
        print(f"   Real detections: {temporal_stats['real_frames']} ({100*temporal_stats['real_frames']/len(results):.1f}%)")
        print(f"   Prediction consistency: {temporal_stats['prediction_consistency']*100:.1f}%")
        print(f"   Average confidence: {temporal_stats['avg_confidence']*100:.2f}%")
        print(f"   Confidence std dev: {temporal_stats['std_confidence']*100:.2f}%")
        
        print("\n" + "="*80)
        print("⚖️  FINAL VERDICT")
        print("="*80)
        
        verdict_symbol = "🔴 DEEPFAKE DETECTED" if final_prediction == 'FAKE' else "🟢 APPEARS AUTHENTIC"
        print(f"\n   {verdict_symbol}")
        print(f"   Confidence: {final_confidence*100:.2f}%")
        print(f"   Consistency: {temporal_stats['prediction_consistency']*100:.1f}%")
        
        if temporal_stats['prediction_consistency'] < 0.7:
            print("\n   ⚠️  WARNING: Low consistency across frames")
            print("      This video shows mixed signals - some frames appear real, others fake.")
            print("      Possible causes:")
            print("      - Partial manipulation (only some frames edited)")
            print("      - Poor quality video")
            print("      - Compression artifacts")
        elif temporal_stats['std_confidence'] > 0.2:
            print("\n   ℹ️  NOTE: Variable confidence across frames")
            print("      Different parts of the video have different manipulation quality.")
        
        print("\n" + "="*80 + "\n")
    
    def save_visualizations(self, results, temporal_stats, output_dir):
        """Save visualization images"""
        print(f"\n💾 Saving visualizations to {output_dir}...")
        
        # 1. Frame grid with heatmaps
        n_frames = len(results)
        n_cols = min(5, n_frames)
        n_rows = (n_frames + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 3, n_rows * 5))
        fig.suptitle('Frame Analysis with Grad-CAM Heatmaps', fontsize=16, fontweight='bold')
        
        for i, r in enumerate(results):
            row = (i // n_cols) * 2
            col = i % n_cols
            
            # Original frame
            ax1 = axes[row, col] if n_rows > 1 else axes[0, col]
            ax1.imshow(r['frame'])
            pred = r['prediction']
            color = 'red' if pred['class'] == 'FAKE' else 'green'
            ax1.set_title(f"Frame {i+1}\n{pred['class']} ({pred['confidence']*100:.1f}%)", 
                         color=color, fontweight='bold')
            ax1.axis('off')
            
            # Heatmap overlay
            ax2 = axes[row + 1, col] if n_rows > 1 else axes[1, col]
            overlay, _ = self.create_heatmap_overlay(r['frame'], r['cam'], alpha=0.5)
            ax2.imshow(overlay)
            ax2.set_title('Attention Heatmap', fontsize=10)
            ax2.axis('off')
        
        # Hide empty subplots
        for i in range(n_frames, n_rows * n_cols):
            row = (i // n_cols) * 2
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].axis('off')
                axes[row + 1, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'frame_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Confidence timeline
        fig, ax = plt.subplots(figsize=(12, 6))
        
        times = [r['time'] for r in results]
        confidences = [r['prediction']['confidence'] for r in results]
        classes = [r['prediction']['class'] for r in results]
        colors = ['red' if c == 'FAKE' else 'green' for c in classes]
        
        ax.plot(times, confidences, 'o-', linewidth=2, markersize=8)
        ax.scatter(times, confidences, c=colors, s=100, zorder=3)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision boundary')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_title('Prediction Confidence Over Time', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_timeline.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Statistics summary
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        labels = ['FAKE', 'REAL']
        sizes = [temporal_stats['fake_frames'], temporal_stats['real_frames']]
        colors_pie = ['#ff6b6b', '#51cf66']
        axes[0].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Frame Classification Distribution', fontsize=12, fontweight='bold')
        
        # Bar chart
        metrics = ['Avg Confidence', 'Consistency', 'Std Dev']
        values = [
            temporal_stats['avg_confidence'],
            temporal_stats['prediction_consistency'],
            temporal_stats['std_confidence']
        ]
        axes[1].bar(metrics, values, color=['#4dabf7', '#51cf66', '#ff6b6b'])
        axes[1].set_ylim([0, 1])
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('Temporal Statistics', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved: frame_analysis.png")
        print(f"   ✓ Saved: confidence_timeline.png")
        print(f"   ✓ Saved: statistics.png")

def main():
    """Interactive mode"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║        INTERACTIVE DEEPFAKE DETECTOR WITH EXPLANATIONS       ║
║                                                              ║
║  🎯 Frame-by-frame analysis                                  ║
║  🔥 Grad-CAM heatmaps (visual attention)                     ║
║  📊 Confidence scores & reasoning                            ║
║  ⏱️  Temporal consistency analysis                           ║
║  💾 Automatic visualization export                           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Paths
    model_path = Path("D:/deepfake_detector_production/models/best_model.pth")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model first!")
        return
    
    # Initialize analyzer
    analyzer = DeepfakeAnalyzer(model_path, device='cuda')
    
    # Interactive loop
    while True:
        print("\n" + "="*80)
        video_path = input("📹 Enter video path (or 'quit' to exit): ").strip()
        
        if video_path.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        video_path = video_path.strip('"').strip("'")  # Remove quotes
        
        if not Path(video_path).exists():
            print(f"❌ File not found: {video_path}")
            continue
        
        try:
            # Analyze
            analyzer.analyze_video(video_path, num_frames=10, save_visualizations=True)
            
            print("\n✅ Analysis complete!")
            print("   Check 'analysis_results' folder for detailed visualizations.")
            
        except Exception as e:
            print(f"\n❌ Error analyzing video: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
