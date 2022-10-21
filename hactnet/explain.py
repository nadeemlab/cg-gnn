"""
Explain a cell graph (CG) prediction using a pretrained CG-GNN and a graph explainer.

As used in:
"Quantifying Explainers of Graph Neural Networks in Computational Pathology", Jaume et al, CVPR, 2021.
"""

from os import makedirs
from os.path import join
from itertools import combinations, compress
from re import sub
from typing import List, Optional, Tuple, Dict, Union

from tqdm import tqdm
from torch import FloatTensor
from torch.cuda import is_available
from dgl import DGLGraph
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import auc
from numpy import (empty, argsort, array, max, concatenate, reshape, histogram, corrcoef, mean,
                   ones, all, unique, sort, ndarray, inf)
from scipy.stats import wasserstein_distance
from scipy.ndimage.filters import uniform_filter1d
from pandas import DataFrame
from bokeh.models import Circle, MultiLine, WheelZoomTool, HoverTool, CustomJS, Select, ColorBar
from bokeh.plotting import figure, from_networkx
from bokeh.transform import linear_cmap
from bokeh.palettes import YlOrRd8
from bokeh.layouts import row
from bokeh.io import output_file, save
from matplotlib.pyplot import plot, title, savefig, legend, clf

from hactnet.util import CellGraphModel
from hactnet.train import infer_with_model

from hactnet.util.interpretability import (BaseExplainer, GraphLRPExplainer, GraphGradCAMExplainer,
                                           GraphGradCAMPPExplainer, GraphPruningExplainer)


IS_CUDA = is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
CONCEPTS = "phenotypes"


class AttributeSeparability:
    def __init__(
        self,
        classes: List[int],
        keep_nuclei: List[int] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ) -> None:
        """
        AttributeSeparability constructor.

        Args:
            classes (List[int]): Classifications.
            keep_nuclei (List[int]): Number of nuclei to retain each time.
                                     Default to [5, 10, 15, 20, 25, 30, 35, 40, 45, 50].
        """

        self.keep_nuclei_list = keep_nuclei
        self.n_keep_nuclei = len(self.keep_nuclei_list)
        self.classes = classes
        self.n_classes = len(self.classes)
        self.class_pairs = list(combinations(self.classes, 2))
        self.n_class_pairs = len(self.class_pairs)

    def process(
        self,
        importance_list: List[ndarray],
        attribute_list: List[ndarray],
        label_list: List[int],
        attribute_names: List[str]
    ) -> Tuple[Dict[Tuple[int, int], Dict[str, float]],
               Dict[int, Dict[int, ndarray]],
               Dict[Tuple[int, int], Dict[int, Tuple[int, float]]]]:
        """
        Derive metrics based on the explainer importance scores and nuclei-level concepts.

        Args:
            importance_list (List[ndarray]): Cell importance scores output by explainers.
            attribute_list (List[ndarray]): Cell-level attributes (later grouped into concepts).
            label_list (List[int]): Labels.
        """

        # 1. extract number of concepts
        n_attrs = attribute_list[0].shape[1]

        # 2. min max normalize the importance scores
        importance_list = self.normalize_node_importance(
            importance_list)

        # 3. extract all the histograms
        all_histograms = self._compute_attr_histograms(
            importance_list, attribute_list, label_list, n_attrs)

        # 4. compute the Wasserstein distance for all the class pairs
        all_distances = self._compute_hist_distances(all_histograms, n_attrs)

        # 5. compute the AUC over the #k: output will be Omega x #c
        # Addition: find the k-value with the max distance
        all_aucs: Dict[Tuple[int, int], Dict[str, float]] = {}
        k_max_dist: Dict[Tuple[int, int], Dict[int, Tuple[int, float]]] = {}
        for class_pair_id in range(self.n_class_pairs):
            all_aucs[self.class_pairs[class_pair_id]] = {}
            k_max_dist[self.class_pairs[class_pair_id]] = {}
            for attr_id in range(n_attrs):
                attr_name = attribute_names[attr_id]
                all_aucs[self.class_pairs[class_pair_id]][attr_name] = auc(
                    array(self.keep_nuclei_list) /
                    max(self.keep_nuclei_list),
                    all_distances[:, class_pair_id, attr_id]
                )

                k_max = self.keep_nuclei_list[0]
                max_dist = all_distances[k_max, class_pair_id, attr_id]
                for i, k in enumerate(self.keep_nuclei_list):
                    dist = all_distances[i, class_pair_id, attr_id]
                    if dist > max_dist:
                        k_max = k
                        max_dist = dist
                k_max_dist[self.class_pairs[class_pair_id]
                           ][attr_id] = (k_max, max_dist)

        return all_aucs, all_histograms, k_max_dist

    def _compute_hist_distances(
        self,
        all_histograms: Dict,
        n_attr: int
    ) -> ndarray:
        """
        Compute all the pair-wise histogram distances.

        Args:
             all_histograms (Dict): all the histograms.
             n_concepts (int): number of concepts.
        """
        all_distances = empty(
            (self.n_keep_nuclei, self.n_class_pairs, n_attr))
        for k_id, k in enumerate(self.keep_nuclei_list):
            omega = 0
            for tx in range(self.n_classes):
                for ty in range(self.n_classes):
                    if tx < ty:
                        for attr_id in range(n_attr):
                            all_distances[k_id, omega, attr_id] = wasserstein_distance(
                                all_histograms[k][tx][attr_id],
                                all_histograms[k][ty][attr_id]
                            )
                        omega += 1
        return all_distances

    def _compute_attr_histograms(
        self,
        importance_list: List[ndarray],
        attribute_list: List[ndarray],
        label_list: List[int],
        n_attrs: int
    ) -> Dict[int, Dict[int, ndarray]]:
        """
        Compute histograms for all the attributes.

        Args:
            importance_list (List[ndarray]): Cell importance scores output by explainers.
            attribute_list (List[ndarray]): Cell-level attributes.
            label_list (List[int]): Labels.
        Returns:
            all_histograms (Dict[int, Dict[int, ndarray]]): Dict with all the histograms
                                                            for each thresh k (as key),
                                                            tumor type (as key) and
                                                            attributes (as np array).
        """
        all_histograms: Dict[int, Dict[int, ndarray]] = {}
        for k in self.keep_nuclei_list:
            all_histograms[k] = {}

            attrs = [c[argsort(s)[-k:]]
                     for c, s in zip(attribute_list, importance_list)]
            attrs = concatenate(attrs, axis=0)  # (#samples x k) x #attrs
            attrs[attrs == inf] = 0  # ensure no weird values in attributes
            attrs = minmax_scale(attrs)
            # #samples x k x #attrs
            attrs = reshape(attrs, (-1, k, n_attrs))
            attrs = list(attrs)

            for t in range(self.n_classes):

                # i. extract the samples of type t
                selected_attrs = [a for l, a in zip(
                    label_list, attrs) if l == t]
                selected_attrs = concatenate(selected_attrs, axis=0)

                # iii. build the histogram for all the attrs (dim = #nuclei x attr_types)
                all_histograms[k][t] = array(
                    [self.build_hist(selected_attrs[:, attr_id])
                     for attr_id in range(selected_attrs.shape[1])]
                )
        return all_histograms

    @staticmethod
    def normalize_node_importance(node_importance: List[ndarray]) -> List[ndarray]:
        """
        Normalize node importance. Min-max normalization on each sample.

        Args:
            node_importance (List[ndarray]): node importance output by an explainer.
        Returns:
            node_importance (List[ndarray]): Normalized node importance.
        """
        node_importance = [minmax_scale(x) for x in node_importance]
        return node_importance

    @staticmethod
    def build_hist(concept_values: ndarray, num_bins: int = 100) -> ndarray:
        """
        Build a 1D histogram using the concept_values.

        Args:
            concept_values (ndarray): All the nuclei-level values for a concept.
            num_bins (int): Number of bins in the histogram. Default to 100.
        Returns:
            hist (ndarray): Histogram
        """
        hist, _ = histogram(
            concept_values, bins=num_bins, range=(0., 1.), density=True)
        return hist


class SeparabilityAggregator:

    def __init__(
        self,
        separability_scores: Dict[Tuple[int, int], Dict[str, float]],
        concept_grouping: Dict[str, List[str]]
    ) -> None:
        """
            SeparabilityAggregator constructor.

        Args:
            separability_score (Dict[Dict][float]): Separability score for all the class pairs
                                                    (as key) and attributes (as key).
        """
        self.separability_scores = self._group_separability_scores(
            separability_scores, concept_grouping)

    def _group_separability_scores(self,
                                   sep_scores: Dict[Tuple[int, int], Dict[str, float]],
                                   concept_grouping: Dict[str, List[str]]
                                   ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Group the individual attribute-wise separability scores according
        to the grouping concept.

        Args:
            sep_scores (Dict[Tuple[int, int], Dict[str, float]]): Separability scores
        Returns:
            grouped_sep_scores (Dict[int, Dict[str, float]]): Grouped separability scores
        """
        grouped_sep_scores: Dict[Tuple[int, int], Dict[str, float]] = {}

        for class_pair, class_pair_val in sep_scores.items():
            grouped_sep_scores[class_pair] = {}
            for concept_key, concept_attrs in concept_grouping.items():
                val = sum([class_pair_val[attr]
                          for attr in concept_attrs]) / len(concept_attrs)
                grouped_sep_scores[class_pair][concept_key] = val
        return grouped_sep_scores

    def compute_max_separability_score(self, risk: ndarray) -> Dict[Union[Tuple[int, int], str], float]:
        """
        Compute maximum separability score for each class pair. Then the
        aggregate max sep score w/ and w/o risk.

        Returns:
            max_sep_score (Dict[Union[Tuple[int, int], str], float]): Maximum separability score.
        """
        max_sep_score: Dict[Union[Tuple[int, int], str], float] = {}
        for class_pair, class_pair_val in self.separability_scores.items():
            max_sep_score[class_pair] = max(
                [val for _, val in class_pair_val.items()])
        max_sep_score['agg_with_risk'] = sum(
            array([val for _, val in max_sep_score.items()]) *
            risk
        )
        max_sep_score['agg'] = sum(
            [val for key, val in max_sep_score.items() if isinstance(key, tuple)])
        return max_sep_score

    def compute_average_separability_score(self, risk: ndarray) -> Dict[Union[Tuple[int, int], str], float]:
        """
        Compute average separability score for each class pair. Then the
        aggregate avg sep score w/ and w/o risk.

        Returns:
            avg_sep_score (Dict[Union[Tuple[int, int], str], float]): Average separability score.
        """
        avg_sep_score: Dict[Union[Tuple[int, int], str], float] = {}
        for class_pair, class_pair_val in self.separability_scores.items():
            avg_sep_score[class_pair] = mean(
                array([val for _, val in class_pair_val.items()]))
        avg_sep_score['agg_with_risk'] = sum(
            array([val for _, val in avg_sep_score.items()]) *
            risk
        )
        avg_sep_score['agg'] = sum(
            [val for key, val in avg_sep_score.items() if isinstance(key, tuple)])
        return avg_sep_score

    def compute_correlation_separability_score(self,
                                               risk: ndarray,
                                               patho_prior: ndarray
                                               ) -> Dict[Union[Tuple[int, int], str], float]:
        """
        Compute correlation separability score between the prior
        and the concept-wise separability scores.

        Returns:
            corr_sep_score (Dict[Union[Tuple[int, int], str], float]): Correlation separability score.
        """
        sep_scores = DataFrame.from_dict(
            self.separability_scores).to_numpy()
        class_pairs = list(self.separability_scores.keys())
        sep_scores = minmax_scale(sep_scores)
        corrs: Dict[Union[Tuple[int, int], str], float] = {}
        for i_class_pair in range(sep_scores.shape[1]):
            corr_sep_score = corrcoef(
                patho_prior[:, i_class_pair], sep_scores[:, i_class_pair])
            corrs[class_pairs[i_class_pair]] = corr_sep_score[1, 0]
        corrs['agg_with_risk'] = sum(
            array([val for _, val in corrs.items()]) *
            risk
        )
        corrs['agg'] = sum(
            [val for key, val in corrs.items() if isinstance(key, tuple)])
        return corrs


def _make_bokeh_graph_plot(g: DGLGraph,
                           feature_names: List[str],
                           cell_graph_name: str,
                           out_directory: str) -> None:
    "Create bokeh interactive graph visualization."

    if 'importance' not in g.ndata:
        raise ValueError(
            'importance scores not yet found. Run calculate_importance_scores first.')

    # Convert to networkx graph for plotting interactive
    gx = g.to_networkx()
    for i in range(g.num_nodes()):
        feats = g.ndata['feat'][i].detach().numpy()
        for j, feat in enumerate(feature_names):
            gx.nodes[i][feat] = feats[j]
        gx.nodes[i]['importance'] = g.ndata['importance'][i].detach().numpy()
        gx.nodes[i]['radius'] = gx.nodes[i]['importance']*10
        gx.nodes[i]['histological_structure'] = g.ndata['histological_structure'][i].detach(
        ).numpy().astype(int).item()

    # Create bokeh plot and prepare to save it to file
    graph_name = cell_graph_name.split('/')[-1]
    output_file(join(out_directory, graph_name + '.html'),
                title=graph_name)
    f = figure(match_aspect=True, tools=[
        'pan', 'wheel_zoom', 'reset'], title='Cell ROI graph')
    f.toolbar.active_scroll = f.select_one(WheelZoomTool)
    mapper = linear_cmap(  # colors nodes according to importance by default
        'importance', palette=YlOrRd8[::-1], low=0, high=1)
    plot = from_networkx(gx, {i_node: dat.detach().numpy()
                              for i_node, dat in enumerate(g.ndata['centroid'])})
    plot.node_renderer.glyph = Circle(
        radius='radius', fill_color=mapper, line_width=.1, fill_alpha=.7)
    plot.edge_renderer.glyph = MultiLine(line_alpha=0.2, line_width=.5)

    # Add color legend to right of plot
    colorbar = ColorBar(color_mapper=mapper['transform'], width=8)
    f.add_layout(colorbar, 'right')

    # Define data that shows when hovering over a node/cell
    hover = HoverTool(
        tooltips="h. structure: @histological_structure", renderers=[plot.node_renderer])
    hover.callback = CustomJS(
        args=dict(hover=hover,
                  source=plot.node_renderer.data_source),
        code='const feats = ["' + '", "'.join(feature_names) + '"]' +
        """
        if (cb_data.index.indices.length > 0) {
            const node_index = cb_data.index.indices[0];
            const tooltips = [['h. structure', '@histological_structure']];
            for (const feat_name of feats) {
                if (source.data[feat_name][node_index]) {
                    tooltips.push([`${feat_name}`, `@${feat_name}`]);
                }   
            }
            hover.tooltips = tooltips;
        }
    """)

    # Add interactive dropdown to change why field nodes are colored by
    color_select = Select(title='Color by feature', value='importance', options=[
        'importance'] + feature_names)
    color_select.js_on_change('value', CustomJS(
        args=dict(source=plot.node_renderer.data_source,
                  cir=plot.node_renderer.glyph),
        code="""
        const field = cb_obj.value;
        cir.fill_color.field = field;
        source.change.emit();
        """)
    )

    # Place components side-by-side and save to file
    layout = row(f, color_select)
    f.renderers.append(plot)
    f.add_tools(hover)
    save(layout)


def calculate_importance(cell_graphs: List[DGLGraph],
                         model: CellGraphModel,
                         explainer_model: str
                         ) -> List[ndarray]:
    "Calculate the importance for all cells in every graph."

    # Define the explainer
    explainer: BaseExplainer
    explainer_model = explainer_model.lower().strip()
    if explainer_model in {'lrp', 'graphlrpexplainer'}:
        explainer = GraphLRPExplainer(model=model)
    elif explainer_model in {'cam', 'gradcam', 'graphgradcamexplainer'}:
        explainer = GraphGradCAMExplainer(model=model)
    elif explainer_model in {'pp', 'campp', 'gradcampp', 'graphgradcamppexplainer'}:
        explainer = GraphGradCAMPPExplainer(model=model)
    elif explainer_model in {'pruning', 'gnn', 'graphpruningexplainer'}:
        explainer = GraphPruningExplainer(model=model)
    else:
        raise ValueError("explainer_model not recognized.")

    # Set model to train so it'll let us do backpropogation.
    # This shouldn't be necessary since we don't want the model to change at all while running the
    # explainer. In fact, it isn't necessary when running the original histocartography code, but
    # in this version of python and torch, it results in a can't-backprop-in-eval error in torch
    # because calculating the weights requires backprop-ing to get the backward_hook.
    # TODO: Fix this.
    model = model.train()

    # Calculate the importance scores for every graph
    importance_scores_by_graph: List[ndarray] = []
    for g in tqdm(cell_graphs):
        importance_scores, _ = explainer.process(g.to(DEVICE))
        assert isinstance(importance_scores, ndarray)
        g.ndata['importance'] = FloatTensor(importance_scores)
        importance_scores_by_graph.append(importance_scores)

    return importance_scores_by_graph


def generate_interactives(cell_graphs: List[DGLGraph],
                          feature_names: List[str],
                          cell_graph_names: List[str],
                          out_directory: str
                          ) -> None:
    "Create bokeh interactive plots for all graphs in the out_directory."
    makedirs(out_directory, exist_ok=True)
    for i_g, g in enumerate(tqdm(cell_graphs)):
        _make_bokeh_graph_plot(
            g, feature_names, cell_graph_names[i_g], out_directory)


def plot_histogram(all_histograms: Dict[int, Dict[int, ndarray]],
                   save_path: str,
                   attr_id: int,
                   attr_name: str,
                   k: int = 25,
                   smoothing=True) -> None:
    "Create histogram for a single attribute."

    x = array(list(range(100)))
    for i, histogram in all_histograms[k].items():
        plot(x, uniform_filter1d(
            histogram[attr_id], size=5) if smoothing else histogram[attr_id], label=f'Class {i}')

    title(attr_name)
    legend()
    savefig(join(save_path, sub(r'[^\w\-_\. ]', '', attr_name) + '.png'))
    clf()


def prune_misclassified_entries(cell_graphs_and_labels: Tuple[List[DGLGraph], List[int]],
                                model: CellGraphModel,
                                attributes: List[ndarray]
                                ) -> Tuple[List[int], List[ndarray]]:
    "Prune misclassified samples from incoming data."
    labels = cell_graphs_and_labels[1]
    inferred, _ = infer_with_model(cell_graphs_and_labels, model)
    correct = (array(labels) == inferred).tolist()
    labels = list(compress(cell_graphs_and_labels[1], correct))
    attributes = list(compress(attributes, correct))
    return labels, attributes


def calculate_separability(importance_scores: List[ndarray],
                           labels: List[int],
                           attributes: List[ndarray],
                           attribute_names: List[str],
                           concept_grouping: Optional[Dict[str,
                                                           List[str]]] = None,
                           risk: Optional[ndarray] = None,
                           patho_prior: Optional[ndarray] = None,
                           out_directory: Optional[str] = None
                           ) -> Tuple[DataFrame, DataFrame, Dict[Tuple[int, int], DataFrame]]:
    "Generate separability scores for each concept."

    assert len(importance_scores) == len(labels) == len(attributes)

    classes = sort(unique(labels)).tolist()
    if max(labels) + 1 != len(classes):
        raise ValueError('Class missing from assigned labels. Ensure that your labels are '
                         'zero-indexed and that at least one example from every class is present '
                         'in your dataset.')

    # Fetch graph concepts and classes/labels
    if risk is None:
        risk = ones(len(classes)) / len(classes)
    else:
        assert len(risk) == len(classes)

    # Compute separability scores
    least_cells = attributes[0].shape[0]
    for graph_attribute in attributes:
        if graph_attribute.shape[0] < least_cells:
            least_cells = graph_attribute.shape[0]
    separability_calculator = AttributeSeparability(
        classes, list(range(1, least_cells, max((1, round(least_cells/100))))))
    separability_scores, all_histograms, k_max_dist = separability_calculator.process(
        importance_list=importance_scores,
        attribute_list=attributes,
        label_list=labels,
        attribute_names=attribute_names
    )

    # Plot histograms
    if out_directory is not None:
        makedirs(out_directory, exist_ok=True)
        for i, attribute_name in enumerate(attribute_names):
            plot_histogram(all_histograms, out_directory,
                           i, attribute_name, k=25)

    # Compute final qualitative metrics
    if concept_grouping is None:
        # If not explicitly provided, each attribute will be its own concept
        concept_grouping = {cn: [cn] for cn in attribute_names}
    metric_analyser = SeparabilityAggregator(
        separability_scores, concept_grouping)
    df_aggregated = DataFrame({
        'average': metric_analyser.compute_average_separability_score(risk),
        'maximum': metric_analyser.compute_max_separability_score(risk)
    })
    if patho_prior is not None:
        df_aggregated['correlation'] = metric_analyser.compute_correlation_separability_score(
            risk, patho_prior)
    if all(risk == risk[0]):
        df_aggregated.drop('agg_with_risk', axis=0, inplace=True)

    k_max_dist_dfs: Dict[Tuple[int, int], DataFrame] = {}
    for class_pair, k_data in k_max_dist.items():
        k_max_dist_dfs[class_pair] = DataFrame(
            {'k': [dat[0] for dat in k_data.values()],
             'dist': [dat[1] for dat in k_data.values()]},
            index=[attribute_names[i] for i in k_data.keys()])

    return DataFrame(metric_analyser.separability_scores), df_aggregated, k_max_dist_dfs


def explain_cell_graphs(cell_graphs_and_labels: Tuple[List[DGLGraph], List[int]],
                        model: CellGraphModel,
                        explainer_model: str,
                        attributes: List[ndarray],
                        attribute_names: List[str],
                        prune_misclassified: bool = True,
                        concept_grouping: Optional[Dict[str,
                                                        List[str]]] = None,
                        risk: Optional[ndarray] = None,
                        patho_prior: Optional[ndarray] = None,
                        feature_names: Optional[List[str]] = None,
                        cell_graph_names: Optional[List[str]] = None,
                        out_directory: Optional[str] = None
                        ) -> Tuple[DataFrame, DataFrame, Dict[Tuple[int, int], DataFrame]]:
    "Generate explanations for all the cell graphs."

    cell_graphs = cell_graphs_and_labels[0]
    labels = cell_graphs_and_labels[1]
    importance_scores = calculate_importance(
        cell_graphs, model, explainer_model)
    if (out_directory is not None) and (feature_names is not None) and \
            (cell_graph_names is not None):
        generate_interactives(cell_graphs, feature_names,
                              cell_graph_names, out_directory)
    elif (feature_names is not None) or (cell_graph_names is not None):
        raise ValueError('feature_names, cell_graph_names, and out_directory must all be provided '
                         'to create interactive plots.')
    if prune_misclassified:
        labels, attributes = prune_misclassified_entries(
            cell_graphs_and_labels, model, attributes)
    return calculate_separability(
        importance_scores, labels, attributes, attribute_names, concept_grouping, risk,
        patho_prior, out_directory)
