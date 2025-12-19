"""
Train classifier on hoop-relative trajectory features.

Uses the full Basketball-51 dataset with ball trajectory features.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import argparse


class TrajectoryNet(nn.Module):
    """Simple MLP for trajectory features."""

    def __init__(self, input_dim=18, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)


def train_sklearn_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate sklearn models."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    print("\n" + "="*70)
    print("TRAJECTORY MODEL RESULTS")
    print("="*70)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

        # Validate
        y_val_pred = model.predict(X_val_scaled)
        y_val_prob = model.predict_proba(X_val_scaled)[:, 1]

        # Test
        y_test_pred = model.predict(X_test_scaled)
        y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, y_val_prob)

        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_prob)

        results[name] = {
            'val_acc': val_acc, 'val_f1': val_f1, 'val_auc': val_auc,
            'test_acc': test_acc, 'test_f1': test_f1, 'test_auc': test_auc
        }

        print(f"\n{name}:")
        print(f"  Val:  Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
        print(f"  Test: Acc={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")

    # Feature importance from RF
    rf = models['Random Forest']
    feature_names = [
        'hoop_x', 'hoop_y', 'hoop_detected',
        'ball_hoop_dist_start', 'ball_hoop_dist_end', 'ball_hoop_dist_min', 'ball_hoop_dist_mean',
        'approaching_hoop', 'traj_hoop_angle', 'traj_x', 'traj_y',
        'final_vel_x', 'final_vel_y', 'final_hoop_dist_x', 'final_hoop_dist_y',
        'arc_above_hoop', 'arc_height_vs_hoop', 'ball_detection_rate'
    ]

    if len(feature_names) != X_train.shape[1]:
        feature_names = [f'f{i}' for i in range(X_train.shape[1])]

    importances = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])

    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("="*70)
    for name, imp in importances[:10]:
        print(f"  {name:<25} {imp:.4f}")

    return results


def train_pytorch_model(X_train, y_train, X_val, y_val, X_test, y_test,
                        epochs=100, lr=1e-3, batch_size=32, device='cuda'):
    """Train PyTorch MLP."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val_scaled).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    # Class weights
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    weights = torch.FloatTensor([len(y_train)/(2*n_neg), len(y_train)/(2*n_pos)]).to(device)

    model = TrajectoryNet(input_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)

        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_logits = model(X_test_t)
        test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        test_preds = test_logits.argmax(dim=1).cpu().numpy()

    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)

    print(f"\nPyTorch MLP:")
    print(f"  Val:  Acc={best_val_acc:.4f}")
    print(f"  Test: Acc={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")

    return {'test_acc': test_acc, 'test_f1': test_f1, 'test_auc': test_auc}


def analyze_predictions(model, X_test, y_test, scaler):
    """Analyze model predictions by confidence."""
    X_test_scaled = scaler.transform(X_test)
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    y_preds = model.predict(X_test_scaled)

    print("\n" + "="*70)
    print("CONFIDENCE ANALYSIS (Test Set)")
    print("="*70)

    # High confidence miss predictions
    high_conf_miss = (y_probs < 0.3)
    if high_conf_miss.sum() > 0:
        miss_acc = (y_test[high_conf_miss] == 0).mean()
        print(f"\nHigh-conf MISS predictions (prob < 0.3):")
        print(f"  N={high_conf_miss.sum()}, Actual miss rate: {miss_acc:.1%}")

    # High confidence make predictions
    high_conf_make = (y_probs > 0.7)
    if high_conf_make.sum() > 0:
        make_acc = (y_test[high_conf_make] == 1).mean()
        print(f"\nHigh-conf MAKE predictions (prob > 0.7):")
        print(f"  N={high_conf_make.sum()}, Actual make rate: {make_acc:.1%}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_preds)
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/hoop_features')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load data
    X_train = np.load(f'{args.data_dir}/train_hoop_features.npy')
    y_train = np.load(f'{args.data_dir}/train_labels.npy')
    X_val = np.load(f'{args.data_dir}/val_hoop_features.npy')
    y_val = np.load(f'{args.data_dir}/val_labels.npy')
    X_test = np.load(f'{args.data_dir}/test_hoop_features.npy')
    y_test = np.load(f'{args.data_dir}/test_labels.npy')

    print("="*70)
    print("BASKETBALL SHOT PREDICTION - TRAJECTORY FEATURES")
    print("="*70)
    print(f"\nDataset:")
    print(f"  Train: {len(X_train)} ({(y_train==1).sum()} make, {(y_train==0).sum()} miss)")
    print(f"  Val:   {len(X_val)} ({(y_val==1).sum()} make, {(y_val==0).sum()} miss)")
    print(f"  Test:  {len(X_test)} ({(y_test==1).sum()} make, {(y_test==0).sum()} miss)")
    print(f"\nFeatures: {X_train.shape[1]}")
    print(f"Baseline (always make): {(y_test==1).mean():.1%}")

    # Train sklearn models
    results = train_sklearn_models(X_train, y_train, X_val, y_val, X_test, y_test)

    # Train PyTorch model
    pytorch_results = train_pytorch_model(X_train, y_train, X_val, y_val, X_test, y_test, device=args.device)

    # Analyze predictions
    scaler = StandardScaler()
    scaler.fit(X_train)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(scaler.transform(X_train), y_train)
    analyze_predictions(rf, X_test, y_test, scaler)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    best_model = max(results.items(), key=lambda x: x[1]['test_acc'])
    print(f"\nBest sklearn model: {best_model[0]}")
    print(f"  Test accuracy: {best_model[1]['test_acc']:.1%}")
    print(f"  Test AUC: {best_model[1]['test_auc']:.3f}")


if __name__ == '__main__':
    main()
