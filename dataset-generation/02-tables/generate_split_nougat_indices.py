import dataset_paths
from test_settings import templates, md_test_settings, latex_test_settings_col_widths, latex_test_settings_rows, latex_test_settings_total_width
import subprocess
import os
import shutil

BY_TEMPLATE_DIR = "/work/tesi_czaccagnino/tables/by_template"
BY_MD_ROWS_DIR = "/work/tesi_czaccagnino/tables/by_md_rows"
BY_LATEX_ROWS_DIR = "/work/tesi_czaccagnino/tables/by_latex_rows"
BY_LATEX_COLS_DIR = "/work/tesi_czaccagnino/tables/by_latex_cols"
BY_LATEX_TOTAL_DIR = "/work/tesi_czaccagnino/tables/by_latex_total"
SRC_DIR = "/work/tesi_czaccagnino/tables/aggregate/out/arxiv"

BASE_FIRST_CMD = "python /work/tesi_czaccagnino/nougat/nougat/dataset/create_index.py --dir %s --out %s"
BASE_SECOND_CMD = "python /work/tesi_czaccagnino/nougat/nougat/dataset/gen_seek.py %s"

def get_first_cmd(dir) -> str:
    return BASE_FIRST_CMD % (f"{dir}/arxiv", f"{dir}/test.jsonl")

# build aggregate dataset
for template in templates:
    print(template)
    temp_dir = BY_TEMPLATE_DIR+f"/{template}"
    subprocess.run(get_first_cmd(temp_dir).split())
    subprocess.run((BASE_SECOND_CMD % f"{temp_dir}/test.jsonl").split())
    

for i, el in enumerate(md_test_settings):
    md_row_dir = BY_MD_ROWS_DIR+f"/{i}"
    subprocess.run(get_first_cmd(md_row_dir).split())
    subprocess.run((BASE_SECOND_CMD % f"{md_row_dir}/test.jsonl").split())
print("latex")

for cols_i, col_w in enumerate(latex_test_settings_col_widths):
    latex_cols_dir = BY_LATEX_COLS_DIR+f"/{cols_i}"
    subprocess.run(get_first_cmd(latex_cols_dir).split())
    subprocess.run((BASE_SECOND_CMD % f"{latex_cols_dir}/test.jsonl").split())
print("rows")
for rows_i, rows in enumerate(latex_test_settings_rows):
    latex_rows_dir = BY_LATEX_ROWS_DIR+f"/{rows_i}"
    subprocess.run(get_first_cmd(latex_rows_dir).split())
    subprocess.run((BASE_SECOND_CMD % f"{latex_rows_dir}/test.jsonl").split())
print("tw")
for total_i, total_width in enumerate(latex_test_settings_total_width):
    latex_total_dir = BY_LATEX_TOTAL_DIR+f"/{total_i}"
    subprocess.run(get_first_cmd(latex_total_dir).split())
    subprocess.run((BASE_SECOND_CMD % f"{latex_total_dir}/test.jsonl").split())
                

