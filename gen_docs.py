import os
from collections import defaultdict

def nb_ify(path, nb_name):
    base = "https://nbviewer.jupyter.org/github/CarterMcClellan/A.i.Lgorithms/blob/master/{path}/{nb_name}"
    path = path.replace("./", "").replace(" ", "%20") 
    nb_name = nb_name.replace(" ", "%20") 
    return base.format(path=path, nb_name=nb_name)

def set2table(table_name, table, output_path):
    """
    table_name: title of table
    table: list of notebook titles and links
    output_path: write target
    """
    writer = MarkdownTableWriter(
        table_name=table_name,
        headers=["Notebook", "Link"]
        value_matrix=[
            [0,   0.1,      "hoge", True,   0,      "2017-01-01 03:04:05+0900"],
            [2,   "-2.23",  "foo",  False,  None,   "2017-12-23 45:01:23+0900"],
            [3,   0,        "bar",  "true",  "inf", "2017-03-03 33:44:55+0900"],
            [-10, -9.9,     "",     "FALSE", "nan", "2017-01-01 00:00:00+0900"],
        ],
        margin=1  # add a whitespace for both sides of each cell
    )
    writer.write_table(output_path)

if __name__ == "__main__":
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if name.endswith(".ipynb"):
                print(nb_ify(root, name))

