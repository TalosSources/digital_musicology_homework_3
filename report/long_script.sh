pdflatex "\documentclass{article}\usepackage[margin=0.01in]{geometry}\usepackage{pdflscape}\usepackage[dvipsnames]{xcolor}\usepackage{booktabs}\usepackage{multirow}\usepackage{amsmath,amssymb,amsfonts}\usepackage{tikz}\pagestyle{empty}\begin{document}\begin{landscape}\input{$1}\end{landscape}\end{document}"
#pdfcrop texput.pdf || pdf-crop texput.pdf texput-crop.pdf
pdfcropmargins -v -s -p 2 texput.pdf -o texput-crop.pdf
rm texput.log
rm texput.aux
rm texput.pdf
mv texput-crop.pdf "$1.pdf"
