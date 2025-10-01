# Author(s): Dr. Patrick Lemoine

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from mpl_toolkits.mplot3d import Axes3D 



def main(path, name, qSave):
    
    FileLoad = os.path.join(path, name)    

    DATA = pd.read_csv(FileLoad, header=None, decimal='.')
    #LenFile = len(DATA)
    
    scaler = StandardScaler()
    MatDist_scaled = scaler.fit_transform(DATA)
    
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(MatDist_scaled)
    

    plt.figure(figsize=(8,5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_)*100, marker='o')
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.grid()
    
    if qSave:
        base_name = name.rsplit('.', 1)[0]
        output_file = path+"/"+base_name+"_Cumulative_Explained_Variance_PCA.png"
        plt.savefig(output_file)
    
    plt.show()
    

    dist_mat = pdist(DATA)  
    linkage_mat = linkage(dist_mat, method='average')
    
    # Dendrogram
    plt.figure(figsize=(12, 7))
    dendrogram(linkage_mat, leaf_rotation=90)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    
    if qSave:
        base_name = name.rsplit('.', 1)[0]
        output_file = path+"/"+base_name+"_Dendogram.png"
        plt.savefig(output_file)                   
    plt.show()
    

    clusters = fcluster(linkage_mat, 3, criterion='maxclust')
    print("Cluster assignments:", clusters)
    

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_results[:,0], pca_results[:,1], pca_results[:,2], 
                         c=clusters, cmap='jet', s=50)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title(f'3D PCA Scatter Plot - {name}')
    plt.colorbar(scatter, label='Cluster')
    
    if qSave:
        base_name = name.rsplit('.', 1)[0]
        output_file = path+"/"+base_name+"_3D_PCA_ScatterPlot.png"
        plt.savefig(output_file)
        
    plt.show()
    
    # Heatmap 
    plt.figure(figsize=(10, 8))
    sns.heatmap(DATA, cmap='coolwarm', cbar=True)
    plt.title(f"Heatmap of Distance Matrix - {name}")
    
    if qSave:
        base_name = name.rsplit('.', 1)[0]
        output_file = path+"/"+base_name+"_3D_Heatmap.png"
        plt.savefig(output_file)       
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--Path', type=str, default='.', help='Path.')
    parser.add_argument('--Name', type=str, default='Matrix.csv', help='Name.')
    parser.add_argument('--QSave', type=int, default=1, help='QSave.')
    
    args = parser.parse_args()

    main(args.Path, args.Name, args.QSave)
    
    
    