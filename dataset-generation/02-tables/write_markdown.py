import source_text


text_source = source_text.ArxivSource()


def make_md_table(rows):
    """
    rows is a matrix containing the character counts for each cell in the table
    """
    st = text_source.get_words(1)
    st += "\n\n"
    for index, cols in enumerate(rows):
        for i in cols:
           st+=" | "
           st+= text_source.get_words(i)
        st+=" | "
        st+="\n"
        if index == 0:
            for i in cols:
                st+=" | "
                st+= "-"
            st+=" | "
            st+="\n"


    return st



def make_latex_table(rows, col_widths, total_width):
    """
    rows is a matrix containing the character counts for each cell in the table
    """
    widths = [el/sum(col_widths)*total_width for el in col_widths]
    st = text_source.get_words(1)
    st += "\n\n"
    st += r"\begin{tabular}"
    st+='\n'
    st += "{"+"|".join([f"p{{{w:.2f}\linewidth}}" for w in widths])+"}"
    for cols in rows:
        st+= " & ".join([text_source.get_words(i) for i in cols])
        st+=r" \\"+"\n"
    st+=r"\end{tabular}"
    return st

