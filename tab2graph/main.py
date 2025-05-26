import typer

from .cli import (
    fit_gml,
    fit_gml_multi,
    fit_tab,
    tune_gml,
    tune_tab,
    evaluate_gml,
    evaluate_gml_multi, 
    preprocess,
    construct_graph,
    combine_dataset,
    list_builtin,
    download,
    get_node_embed,
    tsne,
    generate_single_task,
)

app = typer.Typer()

app.command()(combine_dataset)
app.command()(fit_gml)
app.command()(fit_gml_multi)
app.command()(fit_tab)
app.command()(tune_gml)
app.command()(tune_tab)
app.command()(evaluate_gml)
app.command()(evaluate_gml_multi)
app.command()(preprocess)
app.command()(construct_graph)
app.command()(list_builtin)
app.command()(download)
app.command()(get_node_embed)
app.command()(tsne)
app.command()(generate_single_task)

if __name__ == '__main__':
    app()
