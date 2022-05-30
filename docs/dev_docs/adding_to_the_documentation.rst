Adding to the documentation
===========================

There is a multitude of ways you can add to the documentation.

Building the docs
-----------------
Run ``tox -e autodoc_generate`` to update the ``*.rst`` files in ``docs/api_docs/``.
If you added a new module/subpackage, a corresponding .rst file will be generated.
If you deleted/renamed one, please delete the outdated file manually.
There is no point in making changes to the .rst files, because they are automatically overwritten frequently anyway

Once this is done, run ``tox -e build_docs`` to build the sphinx docs.
If you see any errors, resolve them and repeat.

Making changes to a notebook
----------------------------
The notebooks in tornadox are synced to markdown files via ``jupytext``, because we need to be able to read ``git diff``'s.
If you make changes to a notebook in ``jupyter``, the corresponding markdown files should be updated automatically.
If you make changes to the ``.ipynb`` file in some editor, run ``jupytext --sync docs/tutorial_notebooks/*.ipynb`` to sync the files again.
Before making a pull request, run the linters on your notebooks via ``tox -e lint`` and check that they execute flawlessly via ``tox -e execute_notebooks``.
