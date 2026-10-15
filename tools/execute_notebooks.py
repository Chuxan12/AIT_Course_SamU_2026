"""Выполнение ноутбуков отдельными ядрами с сохранением результатов."""
import os
import sys
from pathlib import Path

# Ограничение потоков обеспечивает стабильное время на небольших данных.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[name] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/ait-matplotlib'
os.environ['IPYTHONDIR'] = '/tmp/ait-ipython'
os.environ['JUPYTER_RUNTIME_DIR'] = '/tmp/ait-jupyter'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager

root = Path(__file__).resolve().parents[1]
paths = [root / p for p in sys.argv[1:]] or sorted(root.glob('lab*/assignment*.ipynb'))
for path in paths:
    print('Executing', path.relative_to(root), flush=True)
    book = nbformat.read(path, as_version=4)
    manager = AsyncKernelManager(kernel_name='python3')
    # Use this environment's interpreter even without an installed kernelspec.
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(book, km=manager, timeout=1800,
                            resources={'metadata': {'path': str(path.parent)}})
    try:
        client.on_cell_start = lambda cell, cell_index: print("  cell", cell_index, flush=True)
        client.execute(cleanup_kc=True)
    finally:
        nbformat.write(book, path)
    print('Completed', path.name, flush=True)
