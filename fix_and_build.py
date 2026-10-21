import os
import subprocess
import nbformat
from nbclient import NotebookClient

# 1. Run build_lab2_notebook.py
import lab2.build_lab2_notebook as build_lab2_notebook
build_lab2_notebook.create_lab2_notebook()

# 2. Fix LaTeX single backslashes in markdown cells
print("Fixing LaTeX single backslashes in markdown cells...")
nb = nbformat.read('RV-CLT-PS-Lab2.ipynb', as_version=4)

def fix_text(text):
    text = text.replace('\\\\', '<DOUBLE_BS>')
    # Any single \\ was already literal backslash
    # Now <DOUBLE_BS> becomes single backslash \ except inside pmatrix or explicit line breaks
    text = text.replace('<DOUBLE_BS>begin{pmatrix}', '\\begin{pmatrix}')
    text = text.replace('<DOUBLE_BS>end{pmatrix}', '\\end{pmatrix}')
    # Keep \\ in aligned or pmatrix row breaks
    # Replace all remaining <DOUBLE_BS> with single backslash
    text = text.replace('<DOUBLE_BS>', '\\')
    return text

for c in nb.cells:
    if c.cell_type == 'markdown':
        lines = c.source.splitlines(keepends=True)
        new_lines = []
        for l in lines:
            # If line ends with double backslash (LaTeX row break), keep it
            if l.rstrip().endswith('\\\\'):
                content = l.rstrip()[:-2]
                content = content.replace('\\\\', '\\')
                new_lines.append(content + ' \\\\\n')
            else:
                new_lines.append(l.replace('\\\\', '\\'))
        c.source = ''.join(new_lines)

# Re-save notebook
nbformat.write(nb, 'RV-CLT-PS-Lab2.ipynb')
print("Saved cleanly formatted notebook to RV-CLT-PS-Lab2.ipynb")

# 3. Export to HTML
print("Exporting notebook to HTML...")
subprocess.run(['python', '-m', 'nbconvert', '--to', 'html', 'RV-CLT-PS-Lab2.ipynb'], check=True)

# 4. Compile HTML to PDF
print("Compiling HTML to PDF using Microsoft Edge...")
cwd = os.getcwd().replace('\\', '/')
html_path = f'file:///{cwd}/RV-CLT-PS-Lab2.html'
pdf_path = os.path.join(os.getcwd(), 'RV-CLT-PS-Lab2.pdf')

edge_cmd = [
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    '--headless',
    '--disable-gpu',
    '--run-all-compositor-stages-before-draw',
    '--virtual-time-budget=10000',
    '--no-pdf-header-footer',
    f'--print-to-pdf={pdf_path}',
    html_path
]
subprocess.run(edge_cmd, check=True)
print(f"PDF successfully created: {pdf_path} (size: {os.path.getsize(pdf_path)} bytes)")

