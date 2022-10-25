"""
Create and save interactive plots.
"""

from os import makedirs
from os.path import join
from typing import List

from tqdm import tqdm
from dgl import DGLGraph

from bokeh.models import Circle, MultiLine, WheelZoomTool, HoverTool, CustomJS, Select, ColorBar
from bokeh.plotting import figure, from_networkx
from bokeh.transform import linear_cmap
from bokeh.palettes import YlOrRd8
from bokeh.layouts import row
from bokeh.io import output_file, save


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
