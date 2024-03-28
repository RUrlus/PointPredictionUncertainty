# Documentation template with mkdocs
[[_TOC_]]

## Building the documentation

The template enables generating the documentation in HTML and PDF format.
Before building the documentation, you need to install the required
dependencies:

```shell
brew install pandoc
```
or equivalent if you're not on MacOS.

```shell
pip install mkdocs \
    mkdocs-material \
    'mkdocstrings[python]' \
    mkdocs-print-site-plugin \
    mkdocs-jupyter \
    mkdocs-bibtex
```
### HTML version

[mkdocs](https://github.com/mkdocs/mkdocs) comes with a built-in dev-server
that lets you preview your documentation as you work on it. Make sure you're in
the same directory as the `mkdocs.yml` configuration file, and then start the
server by running the command:

```bash
mkdocs serve
```

Open up the URL printed by the command (default `http://127.0.0.1:8000/`) in
your browser, and you'll see the default home page being displayed.

#### Building the site

In order to deploy the HTML documentation, you need to build it with:

```bash
mkdocs build
```

This will create a new directory, named `site` ready to be deployed. Simply
upload the contents of the entire site directory to wherever you're hosting
your website from and you're done.

### PDF version

A PDF can be generated from the browser by navigating to `/print_page/` or
`print_page.html` and printing the page as PDF using the browser.

### API documentation generation

`mkdocs` uses the package [mkdocstrings](https://mkdocstrings.github.io/) to
generate the API documentation. However, you have to configure it in the file
`mkdocs.yml`.

For automatically generating API documentation, see the section
[Automatic code reference
pages](https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages)
from the official the documentation.

### Adding notebooks to the documentation

This template includes the
[mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter) extension,
allowing you to add `jupyter notebooks` out-of-the-box by adding the `.ipynb`
file to the `mkdocs.yml` file.
