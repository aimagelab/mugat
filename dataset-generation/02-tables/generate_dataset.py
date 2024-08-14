from write_markdown import make_md_table, make_latex_table
import dataset_paths
import os
import tempfile
import subprocess
from test_settings import templates, md_test_settings, latex_test_settings_col_widths, latex_test_settings_rows, latex_test_settings_total_width
TO_REPLACE= r"""}}
  >{\raggedright\arraybackslash}"""

REPLACE_WITH=r"""}}
  |"""

def replace_in_file(file_path, table):
    

    with open(file_path, 'r') as file:
        tex_string = file.read()

    to_replace = tex_string.split(r"\begin{document}")[1].split(r"\end{document}")[0]
    tex_string = tex_string.replace(to_replace, table)
    # tex_string = tex_string.replace("longtable", "tabular")
    # tex_string = tex_string.replace(TO_REPLACE, REPLACE_WITH)
    # tex_string = tex_string.replace(r">{\raggedright\arraybackslash}", "")
    with open(file_path, 'w') as file:
        file.write(tex_string)


for template in templates:
    # for i, el in enumerate(md_test_settings):
    #     (tex_path, pdf_path) = dataset_paths.get_md_path(i, template)
    #     par = os.path.split(tex_path)[0]
    #     os.makedirs(par, exist_ok=True)
        
    #     data = make_md_table(el)

    #     with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp:
    #         temp.write(data)
    #         gen_pdf_command = f"pandoc -s {tex_path} -f latex --output={pdf_path}"
    #         gen_tex_command  = f"pandoc -s {temp.name} -f markdown --output={tex_path}"
    #         if template != "base":
    #             #gen_pdf_command+=f" --pdf-engine=xelatex --template {os.path.join(dataset_paths.TEMPLATES_DIR, template)}.tex"
    #             gen_tex_command+=f" --pdf-engine=xelatex --template {os.path.join(dataset_paths.TEMPLATES_DIR, template)}.tex"
    #         temp.close()
    #         print(gen_pdf_command)
    #         subprocess.run(gen_tex_command.split())
    #         replace_in_file(tex_path)
    #         os.unlink(temp.name)

    #     subprocess.run(gen_pdf_command.split())


    for cols_i, col_w in enumerate(latex_test_settings_col_widths):
        for rows_i, rows in enumerate(latex_test_settings_rows):
            for total_i, total_width in enumerate(latex_test_settings_total_width):
                (tex_path, pdf_path) = dataset_paths.get_latex_path(rows_i, cols_i, total_i, template)

                par = os.path.split(tex_path)[0]
                os.makedirs(par, exist_ok=True)

                data = make_latex_table(rows, col_w, total_width)

                with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp:
                    temp.write(data)
                    # gen_pdf_command = f"pandoc -s -f latex {tex_path} --output={pdf_path}"
                    pdf_dir_path = "/".join(pdf_path.split("/")[0:-1])
                    gen_pdf_command = f"pdflatex -halt-on-error -output-directory {pdf_dir_path} {tex_path}"
                    gen_tex_command  = f"pandoc -s -f latex {temp.name} --output={tex_path}"
                    if template != "base":
                        #gen_pdf_command+=f" --pdf-engine=xelatex --template {os.path.join(dataset_paths.TEMPLATES_DIR, template)}.tex"
                        gen_tex_command+=f" --pdf-engine=xelatex --template {os.path.join(dataset_paths.TEMPLATES_DIR, template)}.tex"
                    temp.close()
                    print(gen_pdf_command)
                    subprocess.run(gen_tex_command.split())

                    os.unlink(temp.name)
                replace_in_file(tex_path, data)
                subprocess.run(gen_pdf_command.split())
                


    
