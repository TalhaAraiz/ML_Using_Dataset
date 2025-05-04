import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import confusion_matrix
from plotly.subplots import make_subplots

def plot_missing_values(missing_data):
    """
    Plot missing values in the dataset
    
    Parameters:
    -----------
    missing_data: pd.DataFrame
        DataFrame containing missing value statistics
    """
    if missing_data is None or missing_data.empty:
        st.info("No missing values found in the dataset.")
        return
        
    try:
        fig = px.bar(
            missing_data, 
            x=missing_data.index, 
            y='Percentage (%)',
            title='Missing Values by Column',
            labels={'index': 'Column', 'value': 'Percentage (%)'},
            color='Percentage (%)',
            color_continuous_scale='blues'
        )
        
        fig.update_layout(
            xaxis_title='Column', 
            yaxis_title='Missing Values (%)',
            height=400,
            width=600
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting missing values: {str(e)}")

def plot_correlation_matrix(data):
    """
    Plot correlation matrix for numeric columns
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
    """
    if data is None or data.empty:
        st.error("No data available for correlation matrix.")
        return
        
    try:
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            st.info("Not enough numeric columns to compute correlations.")
            return
        
        # Compute correlation matrix
        corr = numeric_data.corr()
        
        # Create heatmap with Plotly
        fig = px.imshow(
            corr, 
            text_auto=True, 
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title='Correlation Matrix'
        )
        
        fig.update_layout(
            height=500, 
            width=700,
            xaxis_title='Features',
            yaxis_title='Features'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting correlation matrix: {str(e)}")

def plot_feature_importance(feature_importance_df, title='Feature Importance'):
    """
    Plot feature importance from a model
    
    Parameters:
    -----------
    feature_importance_df: pd.DataFrame
        DataFrame with columns 'Feature' and 'Coefficient'/'Importance'
    title: str
        Plot title
    """
    if feature_importance_df is None or feature_importance_df.empty:
        st.info("No feature importance data available.")
        return
        
    try:
        # Sort features by absolute importance
        if 'Coefficient' in feature_importance_df.columns:
            importance_col = 'Coefficient'
        else:
            importance_col = 'Importance'
        
        feature_importance_df['Abs_Importance'] = abs(feature_importance_df[importance_col])
        feature_importance_df = feature_importance_df.sort_values('Abs_Importance', ascending=False)
        
        # Plot feature importance
        fig = px.bar(
            feature_importance_df, 
            x=importance_col, 
            y='Feature',
            title=title,
            orientation='h',
            color=importance_col,
            color_continuous_scale='RdBu'
        )
        
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            height=400,
            width=600,
            xaxis_title='Importance',
            yaxis_title='Feature'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting feature importance: {str(e)}")

def plot_regression_results(y_test, y_pred, title='Actual vs Predicted Values'):
    """
    Plot regression results comparing actual vs predicted values
    
    Parameters:
    -----------
    y_test: array-like
        Actual values
    y_pred: array-like
        Predicted values
    title: str
        Plot title
    """
    if y_test is None or y_pred is None:
        st.error("No data available for regression results plot.")
        return
        
    try:
        # Create a dataframe with actual and predicted values
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        
        # Create a scatter plot
        fig = px.scatter(
            results_df, 
            x='Actual', 
            y='Predicted',
            title=title,
            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
            opacity=0.7
        )
        
        # Add 45-degree line
        max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
        min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], 
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig.update_layout(
            height=500,
            width=600,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting regression results: {str(e)}")

def plot_confusion_matrix(conf_matrix, title='Confusion Matrix'):
    """
    Plot confusion matrix for classification results
    
    Parameters:
    -----------
    conf_matrix: array-like
        Confusion matrix
    title: str
        Plot title
    """
    if conf_matrix is None:
        st.error("No confusion matrix data available.")
        return
        
    try:
        # Create an annotated heatmap
        fig = px.imshow(
            conf_matrix,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=['Class 0', 'Class 1'],
            y=['Class 0', 'Class 1'],
            color_continuous_scale='blues',
            title=title
        )
        
        fig.update_layout(
            width=500, 
            height=500,
            xaxis_title='Predicted Class',
            yaxis_title='Actual Class'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting confusion matrix: {str(e)}")

def plot_clusters(X, clusters, centers=None, title='K-Means Clustering Results'):
    """
    Plot clustering results in 2D
    
    Parameters:
    -----------
    X: pd.DataFrame
        Feature data (should have at least 2 columns for visualization)
    clusters: array-like
        Cluster assignments
    centers: array-like
        Cluster centers
    title: str
        Plot title
    """
    if X is None or clusters is None:
        st.error("No data available for cluster visualization.")
        return
        
    try:
        # If X has more than 2 columns, use the first two for visualization
        if X.shape[1] > 2:
            viz_cols = X.columns[:2]
            st.info(f"Using only the first two features ({viz_cols[0]} and {viz_cols[1]}) for visualization.")
            plot_data = X[viz_cols].copy()
        else:
            plot_data = X.copy()
        
        # Add cluster labels to the data
        plot_data['Cluster'] = clusters
        
        # Create a scatter plot
        fig = px.scatter(
            plot_data, 
            x=plot_data.columns[0], 
            y=plot_data.columns[1],
            color='Cluster',
            title=title,
            labels={
                plot_data.columns[0]: plot_data.columns[0],
                plot_data.columns[1]: plot_data.columns[1]
            },
            color_continuous_scale='viridis'
        )
        
        # Add cluster centers if provided
        if centers is not None:
            if len(centers) > 0 and len(centers[0]) >= 2:
                center_x = centers[:, 0]
                center_y = centers[:, 1]
                
                fig.add_trace(
                    go.Scatter(
                        x=center_x,
                        y=center_y,
                        mode='markers',
                        marker=dict(
                            color='red',
                            size=12,
                            symbol='x'
                        ),
                        name='Cluster Centers'
                    )
                )
        
        fig.update_layout(
            height=500,
            width=600,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting clusters: {str(e)}")

def plot_elbow_method(k_values, inertia, silhouette=None):
    """
    Plot the elbow method results for K-means
    
    Parameters:
    -----------
    k_values: array-like
        Number of clusters
    inertia: array-like
        Inertia values for each k
    silhouette: array-like
        Silhouette scores for each k
    """
    if k_values is None or inertia is None:
        st.error("No data available for elbow method plot.")
        return
        
    try:
        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Inertia', 'Silhouette Score'))
        
        # Plot inertia
        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=inertia,
                mode='lines+markers',
                name='Inertia'
            ),
            row=1, col=1
        )
        
        # Plot silhouette scores if available
        if silhouette is not None:
            fig.add_trace(
                go.Scatter(
                    x=k_values,
                    y=silhouette,
                    mode='lines+markers',
                    name='Silhouette Score'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=400,
            width=800,
            showlegend=True,
            title_text='Elbow Method Analysis'
        )
        
        fig.update_xaxes(title_text='Number of Clusters', row=1, col=1)
        fig.update_xaxes(title_text='Number of Clusters', row=1, col=2)
        fig.update_yaxes(title_text='Inertia', row=1, col=1)
        fig.update_yaxes(title_text='Silhouette Score', row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting elbow method results: {str(e)}")

def plot_train_test_split(train_size, test_size):
    """
    Plot the train-test split visualization
    
    Parameters:
    -----------
    train_size: float
        Size of the training set
    test_size: float
        Size of the test set
    """
    if train_size is None or test_size is None:
        st.error("No data available for train-test split visualization.")
        return
        
    try:
        sizes = [train_size, test_size]
        labels = ['Training Set', 'Test Set']
        
        fig = px.pie(
            values=sizes,
            names=labels,
            title='Train-Test Split',
            color=labels,
            color_discrete_sequence=['#1f77b4', '#ff7f0e']
        )
        
        fig.update_layout(
            height=400,
            width=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting train-test split: {str(e)}")

def plot_stock_data(data, title="Stock Price History"):
    """
    Plot stock price history
    
    Parameters:
    -----------
    data: pd.DataFrame
        Stock price data
    title: str
        Plot title
    """
    if data is None or data.empty:
        st.error("No data available for stock price visualization.")
        return
        
    try:
        # Create a figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['Close'],
                name="Close Price",
                line=dict(color='#1f77b4')
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['Volume'],
                name="Volume",
                line=dict(color='#ff7f0e')
            ),
            secondary_y=True,
        )
        
        # Add figure title
        fig.update_layout(
            title_text=title,
            height=500,
            width=800
        )
        
        # Set x-axis title
        fig.update_xaxes(title_text="Date")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(title_text="Volume", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting stock data: {str(e)}")

def plot_pca_explained_variance(pca):
    """
    Plot PCA explained variance ratio
    
    Parameters:
    -----------
    pca: PCA
        Fitted PCA object
    """
    if pca is None:
        st.error("No PCA data available for visualization.")
        return
        
    try:
        # Create a bar plot of explained variance ratio
        fig = px.bar(
            x=range(1, len(pca.explained_variance_ratio_) + 1),
            y=pca.explained_variance_ratio_,
            title='PCA Explained Variance Ratio',
            labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'}
        )
        
        # Add cumulative explained variance line
        fig.add_trace(
            go.Scatter(
                x=range(1, len(pca.explained_variance_ratio_) + 1),
                y=np.cumsum(pca.explained_variance_ratio_),
                mode='lines+markers',
                name='Cumulative Explained Variance'
            )
        )
        
        fig.update_layout(
            height=500,
            width=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting PCA explained variance: {str(e)}")
