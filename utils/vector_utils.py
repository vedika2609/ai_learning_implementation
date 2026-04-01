import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ipywidgets import Text, Button, VBox, Output, Layout, HTML
from IPython.display import clear_output,display, HTML
from sklearn.decomposition import PCA
from adjustText import adjust_text

def plot_vectors():
    # Define vectors
    v1 = np.array([1, 2])
    v2 = np.array([1, 1])
    array_v = np.array([[3, 2], [5, 6]])

    # Function to calculate cosine similarity
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    # Function to calculate Euclidean distance
    def euclidean_distance(vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    # Calculate cosine similarities and Euclidean distances
    cos_sim_v1_array_v = [cosine_similarity(v1, av) for av in array_v]
    cos_sim_v2_array_v = [cosine_similarity(v2, av) for av in array_v]
    euc_dist_v1_array_v = [euclidean_distance(v1, av) for av in array_v]
    euc_dist_v2_array_v = [euclidean_distance(v2, av) for av in array_v]

    # Create a single plot for both v1 and v2 with array_v
    plt.figure(figsize=(8, 8))

    # Plot v1 and label
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='#fb5c6b')
    plt.text(v1[0] + 0.1, v1[1], f'v1: {tuple(int(x) for x in v1)}', fontsize=9, color='#191c24')

    # Plot v2 and label
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='#fb5c6b')
    plt.text(v2[0] + 0.1, v2[1], f'v2: {tuple(int(x) for x in v2)}', fontsize=9, color='#191c24')

    # Plot array_v vectors and labels
    for i, av in enumerate(array_v):
        plt.quiver(0, 0, av[0], av[1], angles='xy', scale_units='xy', scale=1, color='#08b4cc')
        plt.text(av[0] + 0.1, av[1], f' av[{i}]: {tuple(int(x) for x in av)}', fontsize=9, color='#191c24')

    # Display cosine similarity and Euclidean distance for both v1 and v2 with av
    y_start = 6.5  # Starting y position
    step = 0.3     # Step to separate each line of text
    for i in range(len(array_v)):
        plt.text(0.5, y_start - (2 * i * step), f'Cos(v1, av[{i}]) = {cos_sim_v1_array_v[i]:.4f}', color='#191c24')
        plt.text(0.5, y_start - ((2 * i + 1) * step), f'Dist(v1, av[{i}]) = {euc_dist_v1_array_v[i]:.4f}', color='#191c24')
        plt.text(3.5, y_start - (2 * i * step), f'Cos(v2, av[{i}]) = {cos_sim_v2_array_v[i]:.4f}', color='#191c24')
        plt.text(3.5, y_start - ((2 * i + 1) * step), f'Dist(v2, av[{i}]) = {euc_dist_v2_array_v[i]:.4f}', color='#191c24')

    plt.scatter(array_v[:, 0], array_v[:, 1], color='#191c24', s=10)
    plt.scatter([v1[0], v2[0]], [v1[1], v2[1]], color='#191c24', s=10)

    plt.xlim(0, 6)
    plt.ylim(0, 7)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title("Cosine Similarity & Euclidean Distance", color="#191c24")
    plt.grid(True)
    plt.tight_layout()

    plt.show()



def display_widget(model):
    # Initial data
    sentences = [
        'apple', 'king', 'queen', 'cellphone', 'car', 'automobile', 'fruit', 'man', 'woman',
        "He spoke softly in class", "He whispered quietly during class", "Her daughter brightened the gloomy day"
    ]
    embeddings = model.encode(sentences)

    # PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Output widget for plots
    output_plot = Output()

    # Define pastel colors
    pastel_colors = ['#fa5c69', '#425469', '#00b1cd', '#f44549', '#b3a7d4', '#cccccc', '#8bd3dd', '#c7d3d4', '#738578']

    def plot_embeddings():
        with output_plot:
            clear_output(wait=True)
            plt.figure(figsize=(12, 8))

            # Update colormap for current entries
            color_count = len(sentences)
            colormap = ListedColormap((pastel_colors * (color_count // len(pastel_colors) + 1))[:color_count])

            texts = []
            for color, (label, (x, y)) in zip(colormap.colors, zip(sentences, embeddings_2d)):
                plt.scatter(x, y, color=color, s=100)
                texts.append(plt.text(x, y, label, fontsize=9, color='black'))

            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))

            plt.title('PCA of Word Embeddings', fontsize=16)
            plt.xlabel('Principal Component 1', fontsize=12)
            plt.ylabel('Principal Component 2', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()

    def on_add_word(change):
        nonlocal embeddings_2d, sentences
        new_word = word_input.value.strip()
        if new_word:
            new_embedding = model.encode([new_word])
            new_embedding_2d = pca.transform(new_embedding)

            # Update list and array
            sentences.append(new_word)
            embeddings_2d = np.vstack([embeddings_2d, new_embedding_2d])

            word_input.value = ''  # Clear input
            plot_embeddings()      # Update plot

    # Custom CSS for pastel-colored button
    display(HTML("""
    <style>
        .custom-button {
            background-color: #fa5c69;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 4px;
        }
        .custom-button:hover {
            background-color: #6b252b;
        }
        .output-plot {
            display: flex;
            justify-content: center;
        }
    </style>
    """))

    # Set up the widgets
    word_input = Text(description="Word/Sentence:", layout=Layout(width='400px'), style={'description_width': 'initial'})
    add_button = Button(description="Add Word or Sentence!", layout=Layout(width='200px'))
    add_button.add_class("custom-button")
    add_button.on_click(on_add_word)

    # Display widgets and initial plot
    display(VBox([word_input, add_button], layout=Layout(margin='10px 0')))
    display(HTML("<div class='output-plot'>"))
    display(VBox([output_plot], layout=Layout(align_items='center', justify_content='center')))
    display(HTML("</div>"))

    # Initial plot display
    plot_embeddings()