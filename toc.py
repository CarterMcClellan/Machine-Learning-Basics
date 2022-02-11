import os, collections, pprint, json
from pytablewriter import MarkdownTableWriter

BASE_URL = {
        # "C" : "https://colab.research.google.com/github/{gh_username}/{repo_name}/blob/{branch}/{path}",
        "N" : "https://nbviewer.org/github/{gh_username}/{repo_name}/blob/{branch}/{path}"
        }

GH_USERNAME = "CarterMcClellan"
REPO_NAME = "Machine-Learning-Basics"
BRANCH = "master"

def format_url(path):
    """ 
        converts path (eg foo/bar) into link which can be used to preview the notebook
        eg (https://nbviewer/github/{gh_username}/{repo_name}/blob/{branch}/foo/bar
    """
    return " ".join(["[{}]({})".format(key, value.format(gh_username=GH_USERNAME, repo_name=REPO_NAME, branch=BRANCH, path=path)) for key, value in BASE_URL.items()])

def create_table(section, paths):
    writer = MarkdownTableWriter(
        table_name=section,
        headers=["open", "title"],
        value_matrix=paths,
        margin=1
    )
    return writer.__str__()

def get_blog_json(sections):
    blog_json = []
    for key in sections:
        value = sections[key][0][0]
        blog_json += [{"title": key, "link": value[value.find("(") + 1:value.find(")")] }]

    with open("blog_json.json", "w") as f:
        json.dump(blog_json, f)

if __name__ == "__main__":
    sections = collections.defaultdict(list)
    
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if name.endswith(".ipynb") and ".ipynb_checkpoints" not in root:
                root = root.replace("./", "").replace(" ", "%20")
                name = name.replace(" ", "%20")
                path = root + "/" + name

                open_link, title = format_url(path), name.replace("%20", " ")
                sections[root.replace("%20", " ")] += [[open_link, title]]
    
    output = ""
    for section in sections:
        output += create_table(section, sections[section])
    
    get_blog_json(sections)

    with open("toc.md", "w") as f:
        f.write(output)

