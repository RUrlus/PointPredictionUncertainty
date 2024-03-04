# Point Prediction Uncertainty

A research package for estimation of point prediction uncertainty estimation

## Installation

```shell
git clone https://github.com/RUrlus/PointPredictionUncertainty
cd PointPredictionUncertainty
pip install -e . -v
```

## Contributing

### Git conventions

We are using a squash-merge strategy, this means that all the commits are squashed (combined) into a single commit during the merge.
This makes it easier to keep a clean history.
Note that pull request should have a title that adheres to the below convention.

**The title should have the following structure:**

```shell
<type>[(optional scope)]: <description>
```

Where `<type>` is one of:

- `build`: Changes that affect the build system (pyproject.toml) or dependencies
- `cicd`: Changes to our CI configuration files and scripts
- `chore`: A task such bumping the version for a release
- `docs`: Documentation only changes
- `feat`: A new feature
- `fix`: A bug fix
- `perf`: A code change that improves performance
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, ...)
- `test`: Changes that affect tests

The `scope` is much more flexible but can be a portion of the code, e.g. `pipelines`, `model` or a specific feature or model variant, e.g. `random-forest`

For example:

```shell
fix(dags): Correct dependency of nodes

Node `charlie` depended on `beta` instead of on `alpha`
```

**Please make sure to remove the prefix from the squash commit message**.
In the `set-autocomplete` the drop-down by select `Customize merge commit message` and make sure that the message follows the title convention.

See [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for details.
The types are based of [angular's types](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines).

## Authors

This package is developed by INGA WBAA
