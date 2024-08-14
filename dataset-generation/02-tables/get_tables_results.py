#!/homes/czaccagnino/.conda/envs/nougat/bin/python
from sys import argv
import pandas as pd
from json import loads as json_loads
import os

import dataset_paths
from test_settings import templates, md_test_settings, latex_test_settings_col_widths, latex_test_settings_rows, latex_test_settings_total_width
import subprocess
import os
import shutil

SMALL = False

OUTFILE = f"/work/tesi_czaccagnino/tables/results-{'small' if SMALL else 'base'}.csv"

BY_TEMPLATE_DIR = "/work/tesi_czaccagnino/tables/by_template"
BY_MD_ROWS_DIR = "/work/tesi_czaccagnino/tables/by_md_rows"
BY_LATEX_ROWS_DIR = "/work/tesi_czaccagnino/tables/by_latex_rows"
BY_LATEX_COLS_DIR = "/work/tesi_czaccagnino/tables/by_latex_cols"
BY_LATEX_TOTAL_DIR = "/work/tesi_czaccagnino/tables/by_latex_total"


list = []

def get_results(dir):
    return os.path.join(dir, "nougat-small.json" if SMALL else "nougat-base.json")

files_mapping = {
    "Aggregate": "/work/tesi_czaccagnino/tables/" + ("aggregate-small.json" if SMALL else "aggregate-base.json"),
}    


# build aggregate dataset
for template in templates:
    print(template)
    temp_dir = BY_TEMPLATE_DIR+f"/{template}"
    res = get_results(temp_dir)
    files_mapping.update({f"template {template}": res})
    
for i, el in enumerate(md_test_settings):
    md_row_dir = BY_MD_ROWS_DIR+f"/{i}"
    res = get_results(md_row_dir)
    files_mapping.update({f"md_rows {el}": res})
print("latex")

for cols_i, col_w in enumerate(latex_test_settings_col_widths):
    latex_cols_dir = BY_LATEX_COLS_DIR+f"/{cols_i}"
    res = get_results(latex_cols_dir)
    files_mapping.update({f"latex_cols {col_w}": res})
print("rows")
for rows_i, rows in enumerate(latex_test_settings_rows):
    latex_rows_dir = BY_LATEX_ROWS_DIR+f"/{rows_i}"
    res = get_results(latex_rows_dir)
    files_mapping.update({f"latex_rows {rows}": res})
print("tw")
for total_i, total_width in enumerate(latex_test_settings_total_width):
    latex_total_dir = BY_LATEX_TOTAL_DIR+f"/{total_i}"
    res = get_results(latex_total_dir)
    files_mapping.update({f"latex_total {total_width}": res})      





for file in files_mapping:
    try:
        with open(files_mapping[file]) as f:
            string = f.read()
            data = json_loads(string)
            list.append({
                "Dataset": file,
                "Size": len(data["predictions"]),
                "Edit distance": data["edit_dist_accuracy"],
                "BLEU": data["bleu_accuracy"],
                "METEOR": data["meteor_accuracy"],
                "Precision": data["precision_accuracy"],
                "Recall": data["recall_accuracy"],
            })
            f.close()
    except:
        print(f"error processing {files_mapping[file]}")
    
df = pd.DataFrame(list)
df.to_csv(OUTFILE)

