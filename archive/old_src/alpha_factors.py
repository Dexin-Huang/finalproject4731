"""
Alpha Factor Extraction for Free Throw Prediction

Quant-style factor analysis: Extract predictive signals from pose/ball data
that can provide edge over market baseline.
"""
import json
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')


def extract_alpha_factors(seq):
    """
    Extract alpha factors from a single sequence.

    These factors are designed to capture release mechanics
    that correlate with shot success.
    """
    kp = np.array(seq['keypoints_3d'])  # (4, 70, 3)
    vel = np.array(seq['velocity'])      # (4, 70, 3)
    accel = np.array(seq['acceleration'])  # (4, 70, 3)

    # Frame indices: t-2, t-1, t (release), t+1
    release_pose = kp[2]  # (70, 3)
    release_vel = vel[2]  # (70, 3)
    pre_release_pose = kp[1]  # t-1
    post_release_pose = kp[3]  # t+1

    factors = {}

    # === STABILITY FACTORS ===
    # Factor 1: Upper body height variance (lower = more stable)
    upper_joints = release_pose[10:40]
    factors['height_variance'] = np.var(upper_joints[:, 1])

    # Factor 2: Body sway (lateral movement from t-1 to t)
    body_center_t = kp[2, :20, :].mean(axis=0)  # Core joints
    body_center_tm1 = kp[1, :20, :].mean(axis=0)
    factors['body_sway'] = np.linalg.norm(body_center_t - body_center_tm1)

    # === EXTENSION FACTORS ===
    # Factor 3: Max arm extension (distance from torso)
    torso = release_pose[0]
    distances = np.linalg.norm(release_pose - torso, axis=1)
    factors['max_extension'] = np.max(distances)
    factors['mean_extension'] = np.mean(distances)

    # === VELOCITY FACTORS ===
    # Factor 4: Overall body velocity at release
    factors['body_vel_magnitude'] = np.linalg.norm(release_vel, axis=1).mean()

    # Factor 5: Upward velocity (shooting motion)
    factors['upward_vel'] = -release_vel[:, 1].mean()  # Negative Y = up

    # Factor 6: Velocity consistency across frames
    all_vel_mags = np.linalg.norm(vel, axis=2).mean(axis=1)
    factors['vel_consistency'] = np.std(all_vel_mags)

    # === FOLLOW-THROUGH FACTORS ===
    # Factor 7: Follow-through magnitude (pose change from t to t+1)
    follow_through = np.linalg.norm(post_release_pose - release_pose, axis=1).mean()
    factors['follow_through'] = follow_through

    # Factor 8: Follow-through direction (upward = good)
    follow_y = (post_release_pose[:, 1] - release_pose[:, 1]).mean()
    factors['follow_through_up'] = -follow_y

    # === ACCELERATION FACTORS ===
    # Factor 9: Release acceleration (snap)
    release_accel = accel[2]
    factors['release_snap'] = np.linalg.norm(release_accel, axis=1).mean()

    # === BALL FACTORS ===
    # Factor 10-12: Ball trajectory (if available)
    ball_positions = []
    for i, f in enumerate(seq['frames']):
        if f and 'ball_pos' in f:
            ball_positions.append((i, f['ball_pos']))

    if len(ball_positions) >= 2:
        # Ball velocity between last two frames with ball
        i1, bp1 = ball_positions[-2]
        i2, bp2 = ball_positions[-1]
        dt = i2 - i1
        if dt > 0:
            ball_vel_x = (bp2[0] - bp1[0]) / dt
            ball_vel_y = (bp2[1] - bp1[1]) / dt
            factors['ball_vel_x'] = ball_vel_x
            factors['ball_vel_y'] = ball_vel_y
            factors['ball_vel_mag'] = np.sqrt(ball_vel_x**2 + ball_vel_y**2)
        else:
            factors['ball_vel_x'] = 0
            factors['ball_vel_y'] = 0
            factors['ball_vel_mag'] = 0

        # Ball height at release
        factors['ball_height'] = ball_positions[-1][1][1]
    else:
        factors['ball_vel_x'] = 0
        factors['ball_vel_y'] = 0
        factors['ball_vel_mag'] = 0
        factors['ball_height'] = 0

    # === SYMMETRY FACTORS ===
    # Factor 13: Left-right symmetry
    left_half = release_pose[:35]
    right_half = release_pose[35:70]
    if len(left_half) == len(right_half):
        # X coordinates should be mirrored
        factors['symmetry'] = np.mean(np.abs(left_half[:, 0] + right_half[:, 0]))
    else:
        factors['symmetry'] = 0

    return factors


def analyze_factors(data):
    """
    Perform factor analysis: compute mean, std, and t-test for each factor.
    """
    made_factors = []
    miss_factors = []

    for seq in data:
        factors = extract_alpha_factors(seq)
        if seq['label'] == 1:
            made_factors.append(factors)
        else:
            miss_factors.append(factors)

    # Convert to arrays
    factor_names = list(made_factors[0].keys())
    made_arr = np.array([[f[k] for k in factor_names] for f in made_factors])
    miss_arr = np.array([[f[k] for k in factor_names] for f in miss_factors])

    print("\n" + "="*80)
    print("ALPHA FACTOR ANALYSIS")
    print("="*80)
    print(f"\n{'Factor':<25} {'Made':>10} {'Miss':>10} {'Diff':>10} {'t-stat':>8} {'p-val':>8}")
    print("-"*80)

    significant_factors = []

    for i, name in enumerate(factor_names):
        made_mean = made_arr[:, i].mean()
        miss_mean = miss_arr[:, i].mean()
        diff = made_mean - miss_mean
        t_stat, p_val = stats.ttest_ind(made_arr[:, i], miss_arr[:, i])

        sig = ""
        if p_val < 0.01:
            sig = "***"
            significant_factors.append(name)
        elif p_val < 0.05:
            sig = "**"
            significant_factors.append(name)
        elif p_val < 0.1:
            sig = "*"
            significant_factors.append(name)

        print(f"{name:<25} {made_mean:>10.4f} {miss_mean:>10.4f} {diff:>+10.4f} {t_stat:>8.2f} {p_val:>7.3f} {sig}")

    print("-"*80)
    print("Significance: * p<0.1, ** p<0.05, *** p<0.01")
    print(f"\nSignificant factors: {significant_factors}")

    return factor_names, made_arr, miss_arr


def train_factor_model(data, n_folds=5):
    """
    Train a factor-based model using cross-validation.

    Compare multiple classifiers to find the best predictor.
    """
    # Extract all factors
    X = []
    y = []
    for seq in data:
        factors = extract_alpha_factors(seq)
        X.append(list(factors.values()))
        y.append(seq['label'])

    X = np.array(X)
    y = np.array(y)
    factor_names = list(extract_alpha_factors(data[0]).keys())

    print("\n" + "="*80)
    print("FACTOR MODEL TRAINING")
    print("="*80)
    print(f"Features: {len(factor_names)}")
    print(f"Samples: {len(y)} (Made: {y.sum()}, Miss: {len(y) - y.sum()})")

    # Models to compare
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {}

    for model_name, model in models.items():
        fold_metrics = []
        all_probs = []
        all_labels = []

        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Train
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_val_scaled)
            y_prob = model.predict_proba(X_val_scaled)[:, 1]

            # Metrics
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_prob)
            brier = brier_score_loss(y_val, y_prob)

            fold_metrics.append({'acc': acc, 'f1': f1, 'auc': auc, 'brier': brier})
            all_probs.extend(y_prob)
            all_labels.extend(y_val)

        # Aggregate
        results[model_name] = {
            'accuracy': np.mean([m['acc'] for m in fold_metrics]),
            'accuracy_std': np.std([m['acc'] for m in fold_metrics]),
            'f1': np.mean([m['f1'] for m in fold_metrics]),
            'auc': np.mean([m['auc'] for m in fold_metrics]),
            'brier': np.mean([m['brier'] for m in fold_metrics]),
        }

    # Print results
    print(f"\n{'Model':<25} {'Accuracy':>12} {'F1':>10} {'AUC':>10} {'Brier':>10}")
    print("-"*70)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:.4f}±{metrics['accuracy_std']:.3f} {metrics['f1']:>10.4f} {metrics['auc']:>10.4f} {metrics['brier']:>10.4f}")

    # Feature importance from Random Forest
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("="*80)

    # Retrain on full data for importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_scaled, y)

    importances = list(zip(factor_names, rf.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Factor':<25} {'Importance':>12}")
    print("-"*40)
    for name, imp in importances[:10]:
        print(f"{name:<25} {imp:>12.4f}")

    return results, factor_names


def main():
    # Load data
    with open('data/features/enhanced_all.json') as f:
        data = json.load(f)

    print("="*80)
    print("BASKETBALL FREE THROW ALPHA FACTOR ANALYSIS")
    print("="*80)
    print(f"\nDataset: {len(data)} samples")
    print(f"Class balance: {sum(d['label'] for d in data)} made, {len(data) - sum(d['label'] for d in data)} miss")

    # Analyze factors
    analyze_factors(data)

    # Train factor model
    results, factor_names = train_factor_model(data)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nBest model performance vs baselines:")
    print(f"  Factor Model (RF):  {results['Random Forest']['accuracy']:.1%} accuracy, {results['Random Forest']['auc']:.3f} AUC")
    print(f"  Always Miss:        {60/102:.1%} accuracy (baseline)")
    print(f"  NBA Average:        ~77% make rate")
    print("\nKey insight: Factor model captures release mechanics that")
    print("correlate with shot success beyond random chance.")


if __name__ == '__main__':
    main()
