from PyFlow.Core import NodeBase
from PyFlow.Core.Common import *
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Packages.PyFlowBase.Pins.StringPin import StringPin
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from PySide2.QtWidgets import QLabel, QInputDialog, QDialog, QVBoxLayout, QLineEdit, QPushButton, QMessageBox
from PySide2.QtCore import Qt
import numpy as np
from sklearn.metrics import silhouette_score
from matplotlib.table import table
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import itertools
from sklearn.decomposition import PCA

class Clustering(NodeBase):
    def __init__(self, name):
        self.data_refreshed = False
        # self.parametersSet = False
        self.messagesShown = False

        ####################### prompt only on refreshing this node
        async def refresh_node(node):
            if node.type == 'Clustering':
                # Show the prompt
                await show_prompt()

            # Refresh the node
            await node.refresh()

            # Update the node's status
            node.status(fill='green', shape='dot', text='refreshed')

        #######################

        super(Clustering, self).__init__(name)

        # Define the input and output pins
        self.dataset = self.createInputPin("Original Dataset", 'AnyPin')
        self.K_plot_wcss = self.createOutputPin("WCSS method (K) plot", 'StringPin')
        self.K_plot_silhouette = self.createOutputPin("Silhouette method (K) plot", 'StringPin')
        self.Best_K = self.createOutputPin("Optimal Number of Clusters (K)", 'StringPin')
        self.document_length = self.createOutputPin("Length of the document", 'StringPin')
        self.elbow_scores_plot = self.createOutputPin("WCSS method (Scores) plot", 'StringPin')
        self.silhouette_scores_plot = self.createOutputPin("Silhouette method (Scores) plot", 'StringPin')

        self.dataset.enableOptions(PinOptions.AllowAny)

    def promptVariables(self):

        if not self.messagesShown:
            # Information dialog
            info_dialog = QMessageBox()
            info_dialog.setWindowTitle("Information")
            info_dialog.setText("Executing... Please wait for the node's green outline to appear.")
            info_dialog.setStandardButtons(QMessageBox.Ok)
            info_dialog.exec_()
            # Set the flag to True after showing the message
            self.messagesShown = True
        else:
            return


    def compute(self):
        self.promptVariables()

        if self.dataset is None or not self.dataset.hasConnections():
            print("No dataset input provided.")
            return

        data = self.dataset.getData()
        if isinstance(data, tuple) and len(data) == 2:
            x_data, y_data = data
        else:
            print("Invalid dataset input format.")
            return

        if isinstance(x_data[0], str):
            # Textual data
            x_data_transformed = preprocess_and_transform(x_data)

            # Perform elbow method to determine the optimal number of clusters (K) based on WCSS
            best_k_wcss, wcss = elbow_method(x_data_transformed)

            # Perform silhouette method to determine the optimal number of clusters (K)
            best_k_silhouette, silhouette_scores = silhouette_method(x_data_transformed)

            # Perform k-means clustering with the best K from WCSS
            kmeans_wcss = KMeans(n_clusters=best_k_wcss)
            kmeans_wcss.fit(x_data_transformed)
            labels_wcss = kmeans_wcss.labels_

            # Generate scatter plot for WCSS method
            scatter_plot_path_wcss = generate_scatter_plot_w(x_data_transformed, labels_wcss, best_k_wcss)

            # Perform k-means clustering with the best K from silhouette method
            kmeans_silhouette = KMeans(n_clusters=best_k_silhouette)
            kmeans_silhouette.fit(x_data_transformed)
            labels_silhouette = kmeans_silhouette.labels_

            # Generate scatter plot for silhouette method
            scatter_plot_path_silhouette = generate_scatter_plot_sil(x_data_transformed, labels_silhouette,
                                                                     best_k_silhouette)

            # Set the scatter plot outputs
            self.K_plot_wcss.setData(scatter_plot_path_wcss)
            self.K_plot_silhouette.setData(scatter_plot_path_silhouette)

            # Print the best K values and scores
            print("Best K (WCSS method):", best_k_wcss)
            print("Best K (Silhouette method):", best_k_silhouette)
            print("WCSS:", wcss)
            print("Silhouette Scores:", silhouette_scores)

            # Scores
            score = pd.DataFrame({
                'Best K (WCSS)': [best_k_wcss],
                'Best K (Silhouette)': [best_k_silhouette]
            })

            # Increase the figure size to accommodate the table
            plt.figure(figsize=(8, 6))

            # Create the table and adjust font size and cell padding
            table12 = plt.table(cellText=score.values, colLabels=score.columns, cellLoc='center', loc='center')

            # Adjust font size
            table12.set_fontsize(18)

            # Color the cells in the first row with light blue
            for cell in table12.get_celld().values():
                if cell.get_text().get_text() in score.columns:
                    cell.set_facecolor('lightblue')

            # Enable content wrapping in cells
            for cell in table12.get_celld().values():
                cell.set_text_props(wrap=True)

            # Adjust cell size
            table12.scale(1, 4.5)  # Increase the cell size by a factor of 1.5

            # Remove axis and spines
            plt.axis('off')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)

            plt.savefig('scores.png')
            plt.close()

            self.Best_K.setData('scores.png')

            # Calculate the length of documents
            document_lengths = [len(doc) for doc in x_data]

            # Create a dictionary to store the label and corresponding document lengths
            data = {'Label': y_data, 'Document Length': document_lengths}

            # Create a DataFrame from the data
            df = pd.DataFrame(data)

            # Calculate the interquartile range (IQR)
            q25 = np.percentile(df['Document Length'], 25)
            q75 = np.percentile(df['Document Length'], 75)
            iqr = q75 - q25

            # Define the lower and upper bounds to exclude outliers
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            # Filter the data to exclude outliers
            df_no_outliers = df[(df['Document Length'] >= lower_bound) & (df['Document Length'] <= upper_bound)]

            # Set plot size and color
            plt.figure(figsize=(10, 6))
            sns.set_palette("Set1")

            # Generate bar plot
            ax = sns.barplot(x='Label', y='Document Length', data=df_no_outliers, linewidth=0.5, saturation=0.8)

            # Set labels and title
            plt.xlabel('Class', fontsize=16)
            plt.ylabel('Document Length', fontsize=16)
            plt.title('Document Length by Class', fontsize=18)

            # Set font size for tick labels
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            # Save the bar plot
            document_length = 'document_length.png'
            plt.savefig(document_length)
            plt.close()

            # Set the bar plot output
            self.document_length.setData(document_length)

            # Plot the WCSS scores - THE ELBOW
            plt.plot(range(2, len(wcss) + 2), wcss, marker='o', linestyle='-')
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
            plt.title("Elbow Method")

            # Mark the optimal K with a red dot
            plt.plot(best_k_wcss, wcss[best_k_wcss - 2], marker='o', markersize=8, color='red')

            # Calculate the range of the WCSS scores
            wcss_range = np.ptp(wcss)

            # Set the padding percentage
            padding_percentage = 10

            # Calculate the padding amount
            padding = (padding_percentage / 100) * wcss_range

            # Set the y-axis limits with padding
            plt.ylim(min(wcss) - padding, max(wcss) + padding)

            # Save the bar plot
            elbow_scores_plot = 'elbow_scores_plot.png'
            plt.savefig(elbow_scores_plot)
            plt.close()

            # Set the bar plot output
            self.elbow_scores_plot.setData(elbow_scores_plot)

            # Plot the silhouette scores
            plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o', linestyle='-')
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Method")
            plt.plot(best_k_silhouette, silhouette_scores[best_k_silhouette - 2], marker='o', markersize=8, color='red')

            # Save the bar plot
            silhouette_scores_plot = 'silhouette_scores_plot.png'
            plt.savefig(silhouette_scores_plot)
            plt.close()

            # Set the bar plot output
            self.silhouette_scores_plot.setData(silhouette_scores_plot)

        else:
            # Numerical data
            x_data_transformed = np.array(x_data)

            # Perform elbow method to determine the optimal number of clusters (K) based on WCSS
            best_k_wcss, wcss = elbow_method(x_data_transformed)

            # Perform silhouette method to determine the optimal number of clusters (K)
            best_k_silhouette, silhouette_scores = silhouette_method(x_data_transformed)

            # Perform k-means clustering with the best K from WCSS
            kmeans_wcss = KMeans(n_clusters=best_k_wcss)
            kmeans_wcss.fit(x_data_transformed)
            labels_wcss = kmeans_wcss.labels_

            # Generate scatter plot for WCSS method
            scatter_plot_path_wcss = generate_scatter_plot_w(x_data_transformed, labels_wcss, best_k_wcss)

            # Perform k-means clustering with the best K from silhouette method
            kmeans_silhouette = KMeans(n_clusters=best_k_silhouette)
            kmeans_silhouette.fit(x_data_transformed)
            labels_silhouette = kmeans_silhouette.labels_

            # Generate scatter plot for silhouette method
            scatter_plot_path_silhouette = generate_scatter_plot_sil(x_data_transformed, labels_silhouette,
                                                                     best_k_silhouette)

            # Set the scatter plot outputs as empty
            self.K_plot_wcss.setData(scatter_plot_path_wcss)
            self.K_plot_silhouette.setData(scatter_plot_path_silhouette)

            # Print the best K values and scores for numerical data
            print("Best K (WCSS method):", best_k_wcss)
            print("Best K (Silhouette method):", best_k_silhouette)
            print("WCSS:", wcss)
            print("Silhouette Scores:", silhouette_scores)

            # Scores
            score = pd.DataFrame({
                'Best K (WCSS)': [best_k_wcss],
                'Best K (Silhouette)': [best_k_silhouette]
            })

            # Increase the figure size to accommodate the table
            plt.figure(figsize=(8, 6))

            # Create the table and adjust font size and cell padding
            table12 = plt.table(cellText=score.values, colLabels=score.columns, cellLoc='center', loc='center')

            # Adjust font size
            table12.set_fontsize(18)

            # Color the cells in the first row with light blue
            for cell in table12.get_celld().values():
                if cell.get_text().get_text() in score.columns:
                    cell.set_facecolor('lightblue')

            # Enable content wrapping in cells
            for cell in table12.get_celld().values():
                cell.set_text_props(wrap=True)

            # Adjust cell size
            table12.scale(1, 4.5)  # Increase the cell size by a factor of 1.5

            # Remove axis and spines
            plt.axis('off')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)

            plt.savefig('scores.png')
            plt.close()

            self.Best_K.setData('scores.png')

            # Plot the WCSS scores - THE ELBOW
            plt.plot(range(2, len(wcss) + 2), wcss, marker='o', linestyle='-')
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
            plt.title("Elbow Method")

            # Mark the optimal K with a red dot
            plt.plot(best_k_wcss, wcss[best_k_wcss - 2], marker='o', markersize=8, color='red')

            # Calculate the range of the WCSS scores
            wcss_range = np.ptp(wcss)

            # Set the padding percentage
            padding_percentage = 10

            # Calculate the padding amount
            padding = (padding_percentage / 100) * wcss_range

            # Set the y-axis limits with padding
            plt.ylim(min(wcss) - padding, max(wcss) + padding)

            # Save the bar plot
            elbow_scores_plot = 'elbow_scores_plot.png'
            plt.savefig(elbow_scores_plot)
            plt.close()

            # Set the bar plot output
            self.elbow_scores_plot.setData(elbow_scores_plot)

            # Plot the silhouette scores
            plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o', linestyle='-')
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Method")
            plt.plot(best_k_silhouette, silhouette_scores[best_k_silhouette - 2], marker='o', markersize=8, color='red')

            # Save the bar plot
            silhouette_scores_plot = 'silhouette_scores_plot.png'
            plt.savefig(silhouette_scores_plot)
            plt.close()

            # Set the bar plot output
            self.silhouette_scores_plot.setData(silhouette_scores_plot)

        # Show a message that the node has finished running
        finish_dialog = QMessageBox()
        finish_dialog.setWindowTitle("Information")
        finish_dialog.setText("Node successfully executed.")
        finish_dialog.setStandardButtons(QMessageBox.Ok)
        finish_dialog.exec_()

    @staticmethod
    def category():
        return '2_Data_Visualization'

def elbow_method(data):
    wcss = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    best_k = find_elbow_point(wcss)
    return best_k, wcss


def find_elbow_point(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 11, wcss[len(wcss) - 1]
    distances = []
    for i in range(len(wcss)):
        x0 = i + 2
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator / denominator)
    return distances.index(max(distances)) + 2


def silhouette_method(data):
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    best_k = silhouette_scores.index(max(silhouette_scores)) + 2
    return best_k, silhouette_scores


def preprocess_and_transform(data):
    vectorizer = TfidfVectorizer()
    x_data_transformed = vectorizer.fit_transform(data)
    return x_data_transformed


def generate_scatter_plot_w(data, labels, k):
    if hasattr(data, 'toarray'):
        data = data.toarray()

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data=principalComponents, columns=['Principal Component 1', 'Principal Component 2'])
    principalDf['Cluster'] = labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=principalDf, x='Principal Component 1', y='Principal Component 2', hue='Cluster', palette='Set1')
    plt.title(f"K-means Clustering (WCSS) - K={k}")
    scatter_plot_path = 'scatter_plot_wcss.png'
    plt.savefig(scatter_plot_path)
    plt.close()
    return scatter_plot_path



def generate_scatter_plot_sil(data, labels, k):
    if hasattr(data, 'toarray'):
        data = data.toarray()

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data=principalComponents, columns=['Principal Component 1', 'Principal Component 2'])
    principalDf['Cluster'] = labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=principalDf, x='Principal Component 1', y='Principal Component 2', hue='Cluster', palette='Set1')
    plt.title(f"K-means Clustering (Silhouette) - K={k}")
    scatter_plot_path = 'scatter_plot_silhouette.png'
    plt.savefig(scatter_plot_path)
    plt.close()
    return scatter_plot_path
