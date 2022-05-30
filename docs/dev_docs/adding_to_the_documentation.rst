Adding to the documentation
===========================

There is a multitude of ways you can add to the documentation.

Building the docs
-----------------
Run ``tox -e build_docs`` to build the sphinx docs.
If you see any errors, resolve them and repeat.
This workflow automatically turns the modules into ``.rst`` files that are then used by sphinx to build the documentation.
It uses ``sphinx-apidoc`` for this purpose.


Making changes to a notebook
----------------------------
The notebooks in tornadox are synced to markdown files via ``jupytext``, because we need to be able to read ``git diff``'s.
If you make changes to a notebook in ``jupyter``, the corresponding markdown files should be updated automatically.
If you make changes to the ``.ipynb`` file in some editor, run ``jupytext --sync docs/tutorial_notebooks/*.ipynb`` to sync the files again.
Before making a pull request, run the linters on your notebooks via ``tox -e lint`` and check that they execute flawlessly via ``tox -e execute_notebooks``.
