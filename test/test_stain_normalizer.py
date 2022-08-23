import numpy as np
import histocartography
from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder, VahadaneStainNormalizer
from PIL import Image

filename = "/nadeem_lab/datasets/BRACS_RoI/previous_versions/Version1_MedIA/Images/val/4_FEA/BRACS_1830_FEA_5.png"
normalizer = VahadaneStainNormalizer(target_path="/nadeem_lab/Carlin/hact-net/data/target.png")

image = np.array(Image.open(filename))

image = normalizer.process(image)
img = Image.fromarray(image,'RGB')
img = img.save('after_A40.png')