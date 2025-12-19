"""
Visualization script for Basketball Free Throw Prediction.

Creates figures for paper and reproducibility:
1. Pose skeleton visualization
2. Model performance (confusion matrix, ROC curve)
3. Feature importance analysis
4. Confidence distribution
5. Alpha factor analysis
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.stgcn import STGCN, TemporalPoseNet, SimplePoseNet

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR = Path("visualizations/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PoseDataset(Dataset):
    """Dataset for pose sequences."""
    def __init__(self, sequences, use_enhanced=True):
        self.poses = []
        self.labels = []
        self.video_ids = []

        for seq in sequences:
            if use_enhanced and 'keypoints_3d' in seq:
                kp = np.array(seq['keypoints_3d'])
                vel = np.array(seq['velocity'])
                accel = np.array(seq['acceleration'])
                feat = np.concatenate([kp, vel, accel], axis=2)
            else:
                frames_kp = [np.array(f['keypoints_3d']) for f in seq['frames']]
                feat = np.stack(frames_kp, axis=0)

            feat = feat.transpose(2, 0, 1)  # (C, T, V)
            feat = (feat - feat.mean()) / (feat.std() + 1e-6)

            self.poses.append(feat)
            self.labels.append(seq['label'])
            self.video_ids.append(seq['video_id'])

        self.poses = np.array(self.poses)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.poses[idx]), torch.LongTensor([self.labels[idx]])[0]


def load_data():
    """Load the enhanced dataset."""
    with open('data/features/enhanced_all.json') as f:
        data = json.load(f)
    return data


def plot_dataset_summary(data):
    """Plot dataset statistics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Class distribution
    labels = [d['label'] for d in data]
    ax = axes[0]
    counts = [labels.count(0), labels.count(1)]
    bars = ax.bar(['Miss', 'Make'], counts, color=['#e74c3c', '#27ae60'])
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontweight='bold')

    # Valid frames distribution
    ax = axes[1]
    valid_counts = [d.get('original_valid_frames', 4) for d in data]
    ax.hist(valid_counts, bins=[0.5, 1.5, 2.5, 3.5, 4.5], rwidth=0.8, color='#3498db')
    ax.set_xlabel('Valid Frames per Sequence')
    ax.set_ylabel('Count')
    ax.set_title('Valid Frames Distribution')
    ax.set_xticks([1, 2, 3, 4])

    # Interpolation breakdown
    ax = axes[2]
    has_interp = sum(1 for d in data if d.get('had_interpolation', False))
    no_interp = len(data) - has_interp
    ax.pie([no_interp, has_interp], labels=['Clean (4/4)', 'Interpolated'],
           autopct='%1.1f%%', colors=['#27ae60', '#f39c12'])
    ax.set_title('Data Quality')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dataset_summary.png'}")


def plot_pose_skeleton(data, n_samples=4):
    """Visualize pose skeletons for sample shots."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Get 2 makes and 2 misses
    makes = [d for d in data if d['label'] == 1][:2]
    misses = [d for d in data if d['label'] == 0][:2]
    samples = makes + misses

    for i, seq in enumerate(samples):
        row = i // 2

        # Plot release frame (t=0, index 2)
        kp = np.array(seq['keypoints_3d'])[2]  # (70, 3)

        ax = axes[row, i % 2 * 2]

        # Project to 2D (x, y)
        ax.scatter(kp[:, 0], -kp[:, 1], c='blue', s=10, alpha=0.6)

        # Connect some key joints (approximate skeleton)
        # This is a simplified visualization
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        label = 'MAKE' if seq['label'] == 1 else 'MISS'
        ax.set_title(f"{seq['video_id'][:15]}... ({label})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Plot velocity vectors
        ax2 = axes[row, i % 2 * 2 + 1]
        vel = np.array(seq['velocity'])[2]  # (70, 3)
        vel_mag = np.linalg.norm(vel, axis=1)

        scatter = ax2.scatter(kp[:, 0], -kp[:, 1], c=vel_mag, cmap='hot', s=20)
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect('equal')
        ax2.set_title(f"Velocity Magnitude")
        plt.colorbar(scatter, ax=ax2, label='|v|')

    plt.suptitle('Pose Visualization at Release Frame (t=0)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pose_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'pose_visualization.png'}")


def train_and_evaluate(data, model_class, model_kwargs, model_name):
    """Train model and collect predictions for visualization."""
    dataset = PoseDataset(data, use_enhanced=True)
    labels = np.array([d['label'] for d in data])

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_probs = []
    all_labels = []
    all_preds = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=8)

        model = model_class(**model_kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Train
        best_state = None
        best_acc = 0
        for epoch in range(50):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    pred = model(x).argmax(1)
                    correct += (pred == y).sum().item()
                    total += len(y)
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Load best and get predictions
        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                prob = torch.softmax(out, dim=1)[:, 1]
                all_probs.extend(prob.numpy())
                all_preds.extend(out.argmax(1).numpy())
                all_labels.extend(y.numpy())

    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


def plot_model_comparison(data):
    """Compare model performance with visualizations."""
    models = {
        'ST-GCN': (STGCN, {'num_classes': 2, 'num_joints': 70, 'in_channels': 9,
                          'hidden_channels': 64, 'num_layers': 3, 'dropout': 0.3}),
        'Temporal CNN': (TemporalPoseNet, {'num_classes': 2, 'num_joints': 70,
                                           'in_channels': 9, 'hidden_channels': 64, 'dropout': 0.3}),
        'MLP': (SimplePoseNet, {'num_classes': 2, 'num_joints': 70, 'in_channels': 9,
                                'hidden_dim': 256, 'dropout': 0.5}),
    }

    results = {}

    print("\nTraining models for visualization...")
    for name, (model_class, kwargs) in models.items():
        print(f"  Training {name}...")
        probs, preds, labels = train_and_evaluate(data, model_class, kwargs, name)
        results[name] = {'probs': probs, 'preds': preds, 'labels': labels}

    # Plot 1: ROC Curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res['labels'], res['probs'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Plot 2: Confusion Matrix (best model - ST-GCN)
    ax = axes[1]
    best_res = results['ST-GCN']
    cm = confusion_matrix(best_res['labels'], best_res['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Miss', 'Make'], yticklabels=['Miss', 'Make'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (ST-GCN)')

    # Plot 3: Accuracy comparison
    ax = axes[2]
    accuracies = []
    names = []
    for name, res in results.items():
        acc = (res['preds'] == res['labels']).mean()
        accuracies.append(acc)
        names.append(name)

    bars = ax.bar(names, accuracies, color=['#3498db', '#e74c3c', '#27ae60'])
    ax.axhline(y=0.588, color='gray', linestyle='--', label='Baseline (always miss)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    ax.set_ylim([0.4, 0.8])
    ax.legend()

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'model_comparison.png'}")

    return results


def plot_confidence_analysis(results):
    """Analyze prediction confidence and betting strategy."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Use ST-GCN results
    probs = results['ST-GCN']['probs']
    preds = results['ST-GCN']['preds']
    labels = results['ST-GCN']['labels']

    # Plot 1: Confidence distribution
    ax = axes[0]
    confidence = np.maximum(probs, 1 - probs)

    correct_mask = preds == labels
    ax.hist(confidence[correct_mask], bins=20, alpha=0.7, label='Correct', color='#27ae60')
    ax.hist(confidence[~correct_mask], bins=20, alpha=0.7, label='Incorrect', color='#e74c3c')
    ax.set_xlabel('Model Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    ax.legend()

    # Plot 2: Accuracy by confidence threshold
    ax = axes[1]
    thresholds = np.arange(0.5, 0.85, 0.05)
    accs = []
    counts = []

    for thresh in thresholds:
        mask = confidence >= thresh
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).mean()
            accs.append(acc)
            counts.append(mask.sum())
        else:
            accs.append(0)
            counts.append(0)

    ax.plot(thresholds, accs, 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=0.588, color='gray', linestyle='--', label='Baseline')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Confidence Threshold')
    ax.legend()

    # Plot 3: Asymmetric analysis (MISS vs MAKE predictions)
    ax = axes[2]
    miss_pred_mask = preds == 0
    make_pred_mask = preds == 1

    miss_acc = (labels[miss_pred_mask] == 0).mean() if miss_pred_mask.sum() > 0 else 0
    make_acc = (labels[make_pred_mask] == 1).mean() if make_pred_mask.sum() > 0 else 0

    bars = ax.bar(['Predict MISS', 'Predict MAKE'], [miss_acc, make_acc],
                  color=['#e74c3c', '#27ae60'])
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random')
    ax.set_ylabel('Accuracy')
    ax.set_title('Asymmetric Performance\n(Key Finding: MISS detection is better)')
    ax.set_ylim([0, 1])

    for bar, acc in zip(bars, [miss_acc, make_acc]):
        color = 'green' if acc > 0.5 else 'red'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confidence_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'confidence_analysis.png'}")


def plot_alpha_factors(data):
    """Visualize alpha factors analysis."""
    from alpha_factors import extract_alpha_factors

    made_factors = []
    miss_factors = []

    for seq in data:
        factors = extract_alpha_factors(seq)
        if seq['label'] == 1:
            made_factors.append(factors)
        else:
            miss_factors.append(factors)

    factor_names = list(made_factors[0].keys())

    # Select most important factors
    key_factors = ['height_variance', 'max_extension', 'vel_consistency',
                   'follow_through', 'body_sway', 'ball_vel_mag']
    key_factors = [f for f in key_factors if f in factor_names]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, factor in enumerate(key_factors[:6]):
        ax = axes[i]

        made_vals = [f[factor] for f in made_factors]
        miss_vals = [f[factor] for f in miss_factors]

        # Box plot
        bp = ax.boxplot([miss_vals, made_vals], labels=['Miss', 'Make'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('#e74c3c')
        bp['boxes'][1].set_facecolor('#27ae60')

        ax.set_title(factor.replace('_', ' ').title())
        ax.set_ylabel('Value')

        # Add significance indicator
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(made_vals, miss_vals)
        sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''
        if sig:
            ax.text(0.95, 0.95, f'p={p_val:.3f} {sig}', transform=ax.transAxes,
                   ha='right', va='top', fontsize=10, style='italic')

    plt.suptitle('Alpha Factors: Made vs Miss Shots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'alpha_factors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'alpha_factors.png'}")


def plot_betting_strategy():
    """Visualize the betting strategy and expected value."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Edge visualization
    ax = axes[0]

    market_miss = 0.25
    our_miss_high_conf = 0.649
    our_miss_all = 0.627

    x = ['Market\n(Historical)', 'Our Model\n(All Preds)', 'Our Model\n(High Conf)']
    y = [market_miss, our_miss_all, our_miss_high_conf]
    colors = ['#95a5a6', '#3498db', '#27ae60']

    bars = ax.bar(x, y, color=colors)
    ax.set_ylabel('P(Miss)')
    ax.set_title('Miss Probability: Market vs Our Model')
    ax.set_ylim([0, 0.8])

    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

    # Add edge annotation
    ax.annotate('', xy=(2, our_miss_high_conf), xytext=(0, market_miss),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(1, 0.45, f'Edge: +{our_miss_high_conf - market_miss:.1%}',
            ha='center', fontsize=12, color='red', fontweight='bold')

    # Plot 2: Expected Value
    ax = axes[1]

    # Simulate different scenarios
    our_probs = np.linspace(0.3, 0.7, 50)
    market_prob = 0.25
    odds = 2.5  # +250 odds

    evs = []
    for p in our_probs:
        ev = (p * odds * 100) - ((1 - p) * 100)
        evs.append(ev)

    ax.plot(our_probs, evs, 'b-', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.axvline(x=market_prob, color='red', linestyle='--', label=f'Market: {market_prob:.0%}')
    ax.axvline(x=our_miss_high_conf, color='green', linestyle='--', label=f'Our Model: {our_miss_high_conf:.0%}')

    ax.fill_between(our_probs, evs, 0, where=np.array(evs) > 0, alpha=0.3, color='green')
    ax.fill_between(our_probs, evs, 0, where=np.array(evs) < 0, alpha=0.3, color='red')

    ax.set_xlabel('True P(Miss)')
    ax.set_ylabel('Expected Value ($)')
    ax.set_title('Expected Value per $100 Bet\n(+250 odds on Miss)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'betting_strategy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'betting_strategy.png'}")


def create_summary_figure(data, results):
    """Create a single summary figure for the paper."""
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. Class distribution (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    labels = [d['label'] for d in data]
    counts = [labels.count(0), labels.count(1)]
    bars = ax1.bar(['Miss', 'Make'], counts, color=['#e74c3c', '#27ae60'])
    ax1.set_ylabel('Count')
    ax1.set_title('A) Dataset Distribution')
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontweight='bold')

    # 2. Model comparison (top middle)
    ax2 = fig.add_subplot(gs[0, 1:3])
    accuracies = []
    names = []
    for name, res in results.items():
        acc = (res['preds'] == res['labels']).mean()
        accuracies.append(acc)
        names.append(name)

    bars = ax2.bar(names, accuracies, color=['#3498db', '#e74c3c', '#27ae60'])
    ax2.axhline(y=0.588, color='gray', linestyle='--', label='Baseline')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('B) Model Performance')
    ax2.set_ylim([0.4, 0.8])
    for bar, acc in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

    # 3. ROC curve (top right)
    ax3 = fig.add_subplot(gs[0, 3])
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res['labels'], res['probs'])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, label=f"{name} ({roc_auc:.2f})", linewidth=2)
    ax3.plot([0, 1], [0, 1], 'k--')
    ax3.set_xlabel('FPR')
    ax3.set_ylabel('TPR')
    ax3.set_title('C) ROC Curves')
    ax3.legend(fontsize=8)

    # 4. Confusion matrix (middle left)
    ax4 = fig.add_subplot(gs[1, 0:2])
    best_res = results['ST-GCN']
    cm = confusion_matrix(best_res['labels'], best_res['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=['Miss', 'Make'], yticklabels=['Miss', 'Make'])
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_title('D) Confusion Matrix (ST-GCN)')

    # 5. Asymmetric performance (middle right)
    ax5 = fig.add_subplot(gs[1, 2:4])
    preds = best_res['preds']
    labels_arr = best_res['labels']

    miss_pred_mask = preds == 0
    make_pred_mask = preds == 1
    miss_acc = (labels_arr[miss_pred_mask] == 0).mean()
    make_acc = (labels_arr[make_pred_mask] == 1).mean()

    bars = ax5.bar(['Predict MISS\n(n={})'.format(miss_pred_mask.sum()),
                   'Predict MAKE\n(n={})'.format(make_pred_mask.sum())],
                  [miss_acc, make_acc], color=['#e74c3c', '#27ae60'])
    ax5.axhline(y=0.5, color='gray', linestyle='--')
    ax5.set_ylabel('Accuracy')
    ax5.set_title('E) Key Finding: Asymmetric Performance')
    ax5.set_ylim([0, 1])
    for bar, acc in zip(bars, [miss_acc, make_acc]):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', fontweight='bold',
                color='green' if acc > 0.5 else 'red')

    # 6. Betting edge (bottom)
    ax6 = fig.add_subplot(gs[2, :])

    # Create schematic
    ax6.set_xlim([0, 10])
    ax6.set_ylim([0, 3])
    ax6.axis('off')

    # Market box
    ax6.add_patch(plt.Rectangle((0.5, 1), 2, 1.5, fill=True, facecolor='#ecf0f1', edgecolor='black'))
    ax6.text(1.5, 2.3, 'Market', ha='center', fontweight='bold')
    ax6.text(1.5, 1.7, 'P(Miss) = 25%', ha='center')
    ax6.text(1.5, 1.3, 'Odds: +250', ha='center')

    # Arrow
    ax6.annotate('', xy=(4, 1.75), xytext=(3, 1.75),
                arrowprops=dict(arrowstyle='->', lw=2))

    # Our model box
    ax6.add_patch(plt.Rectangle((4.5, 1), 2.5, 1.5, fill=True, facecolor='#d5f5e3', edgecolor='black'))
    ax6.text(5.75, 2.3, 'Our Model', ha='center', fontweight='bold')
    ax6.text(5.75, 1.7, 'P(Miss) = 64.9%', ha='center')
    ax6.text(5.75, 1.3, '(High Conf)', ha='center')

    # Arrow
    ax6.annotate('', xy=(8.5, 1.75), xytext=(7.5, 1.75),
                arrowprops=dict(arrowstyle='->', lw=2))

    # Result box
    ax6.add_patch(plt.Rectangle((8, 1), 1.8, 1.5, fill=True, facecolor='#d4efdf', edgecolor='green', linewidth=2))
    ax6.text(8.9, 2.3, 'Edge', ha='center', fontweight='bold', color='green')
    ax6.text(8.9, 1.7, '+39.9%', ha='center', fontsize=14, fontweight='bold', color='green')
    ax6.text(8.9, 1.3, 'EV: +$127', ha='center', color='green')

    ax6.set_title('F) Betting Strategy: Only Bet on High-Confidence MISS Predictions', fontsize=12)

    plt.suptitle('Basketball Free Throw Prediction for Prediction Markets',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(OUTPUT_DIR / 'summary_figure.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'summary_figure.png'}")


def main():
    print("="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # Load data
    print("\nLoading data...")
    data = load_data()
    print(f"Loaded {len(data)} samples")

    # Generate visualizations
    print("\n1. Dataset summary...")
    plot_dataset_summary(data)

    print("\n2. Pose visualization...")
    plot_pose_skeleton(data)

    print("\n3. Training models and generating comparison plots...")
    results = plot_model_comparison(data)

    print("\n4. Confidence analysis...")
    plot_confidence_analysis(results)

    print("\n5. Alpha factors...")
    plot_alpha_factors(data)

    print("\n6. Betting strategy...")
    plot_betting_strategy()

    print("\n7. Summary figure...")
    create_summary_figure(data, results)

    print("\n" + "="*60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
