import json
#
from animated_tsne import *
from distances import *
from text_embedder import TextEmbedder

COT_END_OFFSET = 2

##### Load

def get_cot(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

        # Find where the last empty string is
        delimiter_index = None
        for i, text in enumerate(data):
            if text == "":
                delimiter_index = i

        # If we found a delimiter, only take texts before it
        if delimiter_index is not None:
            return [text for text in data[:delimiter_index + COT_END_OFFSET] if text.strip()]

        # If no delimiter found, return all non-empty texts
        return [text for text in data if text.strip()]

###### Distances

def aggregate_distances(filepaths):

    distances = []

    for filepath in filepaths:
        # get cot for filepath
        texts = get_cot(filepath)
        # get embeddings
        reference_dict = TextEmbedder().get_reference_dictionary(texts)
        # get dict
        visualizer = AnimatedTSNEVisualizer()
        visualizer.from_dict(reference_dict)
        # get distances
        distances_for_text = visualizer.calculate_consecutive_distances(
            metric="cosine",
            normalization="maxunit"
        )
        # parse distances
        distances_parsed = [distance_for_text["distance"] for distance_for_text in distances_for_text]
        # append
        distances.append(distances_parsed)

    plot_normalized_sequences(distances, show_individual=True)



# get CoT
filepath = "data/chains/1.json"
texts = get_cot(filepath)
reference_dict = TextEmbedder().get_reference_dictionary(texts)

# make visualizaer
visualizer = AnimatedTSNEVisualizer()
visualizer.from_dict(reference_dict)

# simple CoT viz
visualizer.create_animation('img/simple_animation.gif', show_line=True)

# distance bars
visualizer.create_distance_animation('img/distance.gif', metric="cosine", normalization="maxunit")

# combined t-SNE and distance bars
visualizer.create_combined_animation('img/dual_animation.gif', show_line=True)

# aggregate distances
#filepaths=["data/chains/1.json", "data/chains/2.json", "data/chains/3.json", "data/chains/4.json", "data/chains/5.json", "data/chains/6.json", "data/chains/7.json", "data/chains/8.json", "data/chains/9.json", "data/chains/10.json"]
#aggregate_distances(filepaths)
