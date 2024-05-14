import os
from pathlib import Path

import numpy as np

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent


def convert_list_to_tikz(measures, filename, start_index=0):
    # if measure number starts from zero -- add one
    if start_index == 0:
        measures = [elem + 1 for elem in measures]

    if isinstance(measures, np.ndarray):
        measures = measures.tolist()

    measures = measures + [87.0]  # make sure end is in list

    script_path = ROOT_PATH / "report" / "long_script.sh"
    file_path = ROOT_PATH / "report" / filename

    code = """
    \\begin{figure}[!hp]
	\centering
	\\resizebox{\columnwidth}{!}{\\begin{tikzpicture}[
			squareblack/.style={rectangle, draw=black!60, fill=white!5, very thick, minimum size=5mm},
			]
			\centering
    \n
    """

    arc_form = "\\draw[black] ({})+(0,0.3) arc (0:180:{});"
    node_form = "\\node[squareblack] ({}) at ({},{}) {};"

    short_by = 1

    height = 0
    shift = 1

    prev_number = 1
    code += node_form.format(
        int(prev_number * 1000),
        prev_number / short_by - shift / short_by,
        height,
        "{" + f"{prev_number:g}" + "}",
    )

    node_position = {}

    node_position[prev_number] = prev_number / short_by
    for number in measures:
        if prev_number == number:
            continue

        code += "\n"
        code += node_form.format(
            int(number * 1000),
            number / short_by - shift / short_by,
            height,
            "{" + f"{number:g}" + "}",
        )

        node_position[number] = number / short_by

        code += "\n"
        code += arc_form.format(
            int(number * 1000), (node_position[number] - node_position[prev_number]) / 2
        )

        prev_number = number

        if number > 30:
            height = -3
            if shift <= 30:
                shift = number
                code += node_form.format(
                    int(number * 1000),
                    number / short_by - shift / short_by,
                    height,
                    "{" + f"{number:g}" + "}",
                )

        if number > 60:
            height = -6
            if shift <= 60:
                shift = number

                code += node_form.format(
                    int(number * 1000),
                    number / short_by - shift / short_by,
                    height,
                    "{" + f"{number:g}" + "}",
                )

    code += "\n"

    code += """
        \\end{tikzpicture}}
    \\end{figure}
    """

    with file_path.open("w") as f:
        f.write(code)

    command = f"bash {str(script_path)} {str(file_path)[:-4]}"

    os.system(command)

    return code


if __name__ == "__main__":
    array = [
        0.0,
        4.0,
        8.0,
        12.0,
        15.5,
        19.75,
        23.0,
        27.0,
        31.0,
        34.0,
        38.0,
        39.0,
        43.0,
        45.0,
        47.0,
        50.0,
        54.0,
        58.0,
        62.0,
        66.0,
        70.0,
        73.0,
        77.0,
        81.0,
        82.0,
        83.0,
        85.0,
    ]
    print(convert_list_to_tikz(array, "example.tex"))
