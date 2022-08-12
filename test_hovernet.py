import numpy as np
import histocartography
from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder, VahadaneStainNormalizer
from PIL import Image

filename = "/nadeem_lab/datasets/BRACS/BRACS_RoI/previous_versions/val/4_FEA/BRACS_1830_FEA_5.png"
nuclei_detector = NucleiExtractor()
normalizer = VahadaneStainNormalizer(target_path="/nadeem_lab/Eliram/repos/hact-net/data/target.png")
feature_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=72)
knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)
image = np.array(Image.open(filename))
image = normalizer.process(image)
nuclei_map, nuclei_centroids = nuclei_detector.process(image)
features = feature_extractor.process(image, nuclei_map)
cell_graph = knn_graph_builder.process(nuclei_map, features)

from histocartography.visualization import OverlayGraphVisualization, InstanceImageVisualization

visualizer = OverlayGraphVisualization(
    instance_visualizer=InstanceImageVisualization(
        instance_style="filled+outline"
    )
)

viz_cg = visualizer.process(
    canvas=image,
    graph=cell_graph,
    instance_map=nuclei_map
)

viz_cg.save('processed_A40.jpg')