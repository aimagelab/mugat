import dataset_paths
from test_settings import templates, md_test_settings, latex_test_settings_col_widths, latex_test_settings_rows, latex_test_settings_total_width
import subprocess
import shutil

OUT_DIR = "/work/tesi_czaccagnino/tables/aggregate"

TEX_DIR = OUT_DIR+"/tex"
PDF_DIR = OUT_DIR+"/pdf"


# build aggregate dataset
for template in templates:
    for i, el in enumerate(md_test_settings):
        (tex_path, pdf_path) = dataset_paths.get_md_path(i, template)
        shutil.copyfile(tex_path, TEX_DIR+f"/{template}-{i}.tex")
        shutil.copyfile(pdf_path, PDF_DIR+f"/{template}-{i}.pdf")


    for cols_i, col_w in enumerate(latex_test_settings_col_widths):
        for rows_i, rows in enumerate(latex_test_settings_rows):
            for total_i, total_width in enumerate(latex_test_settings_total_width):
                (tex_path, pdf_path) = dataset_paths.get_latex_path(rows_i, cols_i, total_i, template)
                shutil.copyfile(tex_path, TEX_DIR+f"/{template}-{cols_i}-{rows_i}-{total_i}.tex")
                shutil.copyfile(pdf_path, PDF_DIR+f"/{template}-{cols_i}-{rows_i}-{total_i}.pdf")

