from write_markdown import make_md_table, make_latex_table
import dataset_paths
import os
import subprocess
import shutil
from test_settings import templates, md_test_settings, latex_test_settings_col_widths, latex_test_settings_rows, latex_test_settings_total_width

HTML_DIR = os.path.join(dataset_paths.BASE_DIR, "html")
NEW_HTML_DIR = os.path.join(os.path.join(dataset_paths.BASE_DIR, "aggregate"), "html")

GOOD_TEMPLATE = "newsprint"

def is_good_filename(st):
    return st.startswith(GOOD_TEMPLATE)

print(os.listdir(HTML_DIR))

good_files = list(filter(is_good_filename, os.listdir(HTML_DIR)))

print(good_files)

for template in templates:
    for file in good_files:
        new_file = file.replace(file.split("-")[0], template)
        print(f"generating {new_file} using {file}")
        shutil.copyfile(os.path.join(HTML_DIR, file), os.path.join(NEW_HTML_DIR, new_file))
    
    
