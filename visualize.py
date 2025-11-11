import pandas as pd
import matplotlib as plt
import seaborn as sns

def visualize_pca_feature_importance(best_model, X_train, y_train, top_n_features=10, top_n_pcs=5):
    
    pca_step_name = [name for name in best_model.named_steps if 'pca' in name.lower()][0]
    pca_step = best_model.named_steps[pca_step_name]
    
    loadings = pd.DataFrame(
        pca_step.components_.T,
        index=X_train.columns,
        columns=[f'PC{i+1}' for i in range(pca_step.n_components_)]
    )
    loadings['importance'] = np.sum(np.abs(loadings), axis=1)
    loadings = loadings.sort_values('importance', ascending=False)
    
    print("Top features by PCA importance:\n", loadings.head(top_n_features))
    
    # --- Transform X_train to PCA space ---
    X_pca = pca_step.transform(X_train)
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(pca_step.n_components_)])
    X_pca_df['label'] = y_train.values
    
    # --- Correlation of PCs with label ---
    corr_with_label = X_pca_df.drop(columns='label').corrwith(X_pca_df['label']).abs().sort_values(ascending=False)
    
    # --- Heatmap of top features × all PCs ---
    top_features = loadings.head(top_n_features).index
    loadings_top_features = loadings.loc[top_features, loadings.columns[:-1]] 
    loadings_scaled = loadings_top_features.apply(lambda x: x / np.max(np.abs(x)), axis=0)
    
    plt.figure(figsize=(min(20, pca_step.n_components_*1.5),6))
    sns.heatmap(loadings_scaled, annot=True, cmap='coolwarm', center=0)
    plt.title(f"PCA Loadings (Top {top_n_features} Features)")
    plt.xlabel("Principal Components")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()
    
    # --- Barplot of top PCs by correlation with label ---
    top_pcs = corr_with_label.head(top_n_pcs).index
    plt.figure(figsize=(8,4))
    sns.barplot(x=top_pcs, y=corr_with_label[top_pcs])
    plt.ylabel("Absolute correlation with label")
    plt.title(f"Top {top_n_pcs} PCs Correlated with Label")
    plt.show()
    
    return loadings, corr_with_label

def plot_curves(results, y_proba, sampling_method):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC-AUC={results['test_roc_auc']:.3f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({sampling_method})")
    plt.legend()
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"AP={results['test_ap']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({sampling_method})")
    plt.legend()
    plt.show()

    # Confusion Matrix
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix {sampling_method})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    print(results['classification_report'])
    return results