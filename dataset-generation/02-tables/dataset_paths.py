import os


TEMPLATES_DIR = "/work/tesi_czaccagnino/Pandoc-Themes"

BASE_DIR = "/work/tesi_czaccagnino/tables/by_separators/sep"

MD_SUBDIR = "md"

LATEX_SUBDIR = "tex"

def get_md_path(setting_idx, template):
    base_path = os.path.join(BASE_DIR, MD_SUBDIR)
    dir = os.path.join(base_path, str(setting_idx))

    return (os.path.join(dir, f"{template}.tex", ), os.path.join(dir, f"{template}.pdf", ))

def get_latex_path(rows_setting_idx, widths_settings_idx, total_width_setting_idx, template):
    base_path = os.path.join(BASE_DIR, LATEX_SUBDIR)

    dir = os.path.join(os.path.join(os.path.join(base_path, str(rows_setting_idx)), str(widths_settings_idx)), str(total_width_setting_idx))

    return (os.path.join(dir, f"{template}.tex", ), os.path.join(dir, f"{template}.pdf", ))
