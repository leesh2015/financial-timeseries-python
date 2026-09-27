"""
Chapter 1: Linear Algebra - Factor Analysis using PCA

Core Analogy: "Extracting the Essential Flavor of a Cocktail"
- Compress hundreds of stock movements into a few key factors like 'market', 'interest rate', 'oil price'
- Eigenvalue decomposition: Finding the main directions in the data

This example demonstrates:
1. Understanding the mathematical principles of PCA (eigenvalue decomposition)
2. Meaning and interpretation of principal components
3. Factor extraction through dimensionality reduction
4. Comparison with Fama-French factor models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def demonstrate_eigenvalue_decomposition():
    """
    1. Basic Principles of Eigenvalue Decomposition
    """
    print("=" * 60)
    print("1. Basic Principles of Eigenvalue Decomposition")
    print("=" * 60)
    
    # Simple covariance matrix example
    cov_matrix = np.array([
        [2.0, 1.0],
        [1.0, 2.0]
    ])
    
    print("\nCovariance Matrix:")
    print(cov_matrix)
    
    # Eigenvalue decomposition: Σ = PΛPᵀ
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    print(f"\nEigenvalues (Λ): {eigenvalues}")
    print("  → Amount of variance explained by each principal component")
    
    print(f"\nEigenvectors (P):")
    print(eigenvectors)
    print("  → Direction of each principal component")
    
    # Verification: PΛPᵀ = Σ
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    print(f"\nReconstructed matrix (PΛPᵀ):")
    print(reconstructed)
    print(f"Difference from original: {np.abs(cov_matrix - reconstructed).max():.10f}")
    
    return eigenvalues, eigenvectors


def load_stock_data(tickers, start_date='2020-01-01', end_date='2024-01-01'):
    """
    2. Load Real Stock Data
    """
    print("\n" + "=" * 60)
    print("2. Load Real Stock Data")
    print("=" * 60)
    
    print(f"\nDownloading {tickers} data...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    returns = data.pct_change().dropna()
    
    print(f"Data period: {returns.index[0].date()} ~ {returns.index[-1].date()}")
    print(f"Number of assets: {len(returns.columns)}")
    print(f"Number of observations: {len(returns)}")
    
    return returns


def perform_pca_analysis(returns, n_components=None):
    """
    3. Perform PCA Analysis
    """
    print("\n" + "=" * 60)
    print("3. Perform PCA Analysis")
    print("=" * 60)
    
    # Data standardization (important!)
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns)
    
    # Perform PCA
    if n_components is None:
        n_components = min(len(returns.columns), len(returns))
    
    pca = PCA(n_components=n_components)
    pca.fit(returns_scaled)
    
    # Transform to principal components
    principal_components = pca.transform(returns_scaled)
    
    print(f"\nNumber of principal components: {len(pca.explained_variance_ratio_)}")
    print(f"\nVariance explained by each principal component:")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:10], 1):
        print(f"  PC{i}: {ratio:.4f} ({ratio*100:.2f}%)")
    
    print(f"\nCumulative explained variance ratio:")
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for i, cum_ratio in enumerate(cumulative[:10], 1):
        print(f"  PC1~PC{i}: {cum_ratio:.4f} ({cum_ratio*100:.2f}%)")
    
    # How many PCs explain most of the variance?
    n_components_90 = np.argmax(cumulative >= 0.90) + 1
    n_components_95 = np.argmax(cumulative >= 0.95) + 1
    
    print(f"\n90% variance explained: PC1~PC{n_components_90}")
    print(f"95% variance explained: PC1~PC{n_components_95}")
    
    return pca, principal_components, returns_scaled


def interpret_principal_components(pca, returns, n_components=5):
    """
    4. Interpret Principal Components
    """
    print("\n" + "=" * 60)
    print("4. Interpret Principal Components")
    print("=" * 60)
    
    # Principal component loadings (how much each asset contributes to each PC)
    loadings = pca.components_[:n_components].T
    
    print(f"\nPrincipal component loadings (contribution of each asset):")
    loadings_df = pd.DataFrame(
        loadings,
        index=returns.columns,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    print(loadings_df)
    
    # Assets contributing most to each PC
    print(f"\nAssets contributing most to each principal component:")
    for i in range(n_components):
        pc_loadings = loadings[:, i]
        top_contributors = np.argsort(np.abs(pc_loadings))[-3:][::-1]
        print(f"  PC{i+1}:")
        for idx in top_contributors:
            print(f"    {returns.columns[idx]}: {pc_loadings[idx]:.4f}")
    
    return loadings_df


def visualize_pca_results(returns, pca, principal_components):
    """
    5. Visualize PCA Results
    """
    print("\n" + "=" * 60)
    print("5. Visualize PCA Results")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Explained variance ratio (Scree Plot)
    ax1 = axes[0, 0]
    explained_variance = pca.explained_variance_ratio_[:10]
    ax1.bar(range(1, len(explained_variance) + 1), explained_variance, 
            color='steelblue', alpha=0.7)
    ax1.set_xlabel('Principal Component Number', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Scree Plot (Explained Variance Ratio)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Cumulative explained variance ratio
    ax2 = axes[0, 1]
    cumulative = np.cumsum(pca.explained_variance_ratio_[:10])
    ax2.plot(range(1, len(cumulative) + 1), cumulative, 
             'o-', color='red', linewidth=2, markersize=8)
    ax2.axhline(y=0.90, color='green', linestyle='--', label='90%')
    ax2.axhline(y=0.95, color='orange', linestyle='--', label='95%')
    ax2.set_xlabel('Number of Principal Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance Ratio', fontsize=12)
    ax2.set_title('Cumulative Explained Variance Ratio', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 2D projection of first two PCs
    ax3 = axes[1, 0]
    if len(principal_components) > 0:
        ax3.scatter(principal_components[:, 0], principal_components[:, 1], 
                   alpha=0.5, s=20, color='blue')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax3.set_title('Principal Component 2D Projection (PC1 vs PC2)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. Principal component loadings heatmap
    ax4 = axes[1, 1]
    loadings = pca.components_[:5].T
    loadings_df = pd.DataFrame(
        loadings,
        index=returns.columns,
        columns=[f'PC{i+1}' for i in range(5)]
    )
    
    im = ax4.imshow(loadings_df.values, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(loadings_df.columns)))
    ax4.set_yticks(range(len(loadings_df.index)))
    ax4.set_xticklabels(loadings_df.columns, rotation=45, ha='right')
    ax4.set_yticklabels(loadings_df.index)
    ax4.set_title('Principal Component Loadings Heatmap', fontsize=14, fontweight='bold')
    
    # Display values
    for i in range(len(loadings_df.index)):
        for j in range(len(loadings_df.columns)):
            text = ax4.text(j, i, f'{loadings_df.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax4)
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'pca_factor_analysis_result.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved: {output_path}")
    plt.show()


def compare_with_fama_french():
    """
    6. Comparison with Fama-French Factors
    """
    print("\n" + "=" * 60)
    print("6. Comparison with Fama-French Factors")
    print("=" * 60)
    
    print("\nPCA Factors vs Fama-French Factors:")
    print("  PCA:")
    print("    - Automatically extracts factors from data")
    print("    - Statistically optimized principal components")
    print("    - May be difficult to interpret")
    print("\n  Fama-French:")
    print("    - Factors with clear economic meaning (market, size, value, etc.)")
    print("    - Has theoretical background")
    print("    - Easy to interpret")
    print("\n  → Using both methods together enables more powerful analysis!")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("Chapter 1: Linear Algebra - Factor Analysis using PCA")
    print("=" * 60)
    
    # 1. Basic principles of eigenvalue decomposition
    eigenvalues, eigenvectors = demonstrate_eigenvalue_decomposition()
    
    # 2. Load real stock data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    returns = load_stock_data(tickers)
    
    # 3. PCA analysis
    pca, principal_components, returns_scaled = perform_pca_analysis(returns)
    
    # 4. Interpret principal components
    loadings_df = interpret_principal_components(pca, returns, n_components=5)
    
    # 5. Visualization
    visualize_pca_results(returns, pca, principal_components)
    
    # 6. Compare with Fama-French
    compare_with_fama_french()
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. PCA finds main directions in data through eigenvalue decomposition")
    print("2. Principal components are directions that explain maximum data variance")
    print("3. Dimensionality reduction: Compress many variables into a few PCs")
    print("4. First principal component (PC1) is usually similar to 'market factor'")
    print("5. Fama-French factors have clear economic meaning, while PCA is data-driven")


if __name__ == "__main__":
    main()

