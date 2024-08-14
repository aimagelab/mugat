import dataset_paths
from test_settings import templates, md_test_settings, latex_test_settings_col_widths, latex_test_settings_rows, latex_test_settings_total_width
import subprocess
import shutil
import os

BY_TEMPLATE_DIR = "/work/tesi_czaccagnino/tables/by_template"
BY_MD_ROWS_DIR = "/work/tesi_czaccagnino/tables/by_md_rows"
BY_LATEX_ROWS_DIR = "/work/tesi_czaccagnino/tables/by_latex_rows"
BY_LATEX_COLS_DIR = "/work/tesi_czaccagnino/tables/by_latex_cols"
BY_LATEX_TOTAL_DIR = "/work/tesi_czaccagnino/tables/by_latex_total"
SRC_DIR = "/work/tesi_czaccagnino/tables/aggregate/out/arxiv"


# build aggregate dataset
for template in templates:
    print(template)
    temp_dir = BY_TEMPLATE_DIR+f"/{template}"
    for i, el in enumerate(md_test_settings):
        md_row_dir = BY_MD_ROWS_DIR+f"/{i}"
        try:
            shutil.copytree(SRC_DIR+f"/{template}-{i}", temp_dir+f"/arxiv/{template}-{i}")
            shutil.copytree(SRC_DIR+f"/{template}-{i}", md_row_dir+f"/arxiv/{template}-{i}")
        except Exception as e:
            print(e)
        
    print("doing latex")

    for cols_i, col_w in enumerate(latex_test_settings_col_widths):
        latex_cols_dir = BY_LATEX_COLS_DIR+f"/{cols_i}"
        for rows_i, rows in enumerate(latex_test_settings_rows):
            latex_rows_dir = BY_LATEX_ROWS_DIR+f"/{rows_i}"
            for total_i, total_width in enumerate(latex_test_settings_total_width):
                latex_total_dir = BY_LATEX_TOTAL_DIR+f"/{total_i}"
                try:
                    shutil.copytree(SRC_DIR+f"/{template}-{cols_i}-{rows_i}-{total_i}", temp_dir+f"/arxiv/{template}-{cols_i}-{rows_i}-{total_i}")
                    shutil.copytree(SRC_DIR+f"/{template}-{cols_i}-{rows_i}-{total_i}", latex_cols_dir+f"/arxiv/{template}-{cols_i}-{rows_i}-{total_i}")
                    shutil.copytree(SRC_DIR+f"/{template}-{cols_i}-{rows_i}-{total_i}", latex_rows_dir+f"/arxiv/{template}-{cols_i}-{rows_i}-{total_i}")
                    shutil.copytree(SRC_DIR+f"/{template}-{cols_i}-{rows_i}-{total_i}", latex_total_dir+f"/arxiv/{template}-{cols_i}-{rows_i}-{total_i}")
                except Exception as e:
                    print(e)
                

