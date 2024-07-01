NAME:='tenprof'
DEV_IMAGE:='ghcr.io/iomorphic/image/dev-py:latest'
SRC_FOLDER:='src'
TEST_FOLDER:='tests'



@default:
    just --list

@verify: lint typecheck test
    echo "Done with Verification"

@pr: init verify
    echo "PR is successful!"

@build:
    pipenv run python -m build

@register:
    git diff --name-only HEAD^1 HEAD -G"^version" "pyproject.toml" | uniq | xargs -I {} sh -c 'just _register'

@_register: init build
    pipenv run twine upload -u $PYPI_USERNAME -p $PYPI_PASSWORD dist/*

@init:
    [ -f Pipfile.lock ] && echo "Lockfile already exists" || PIPENV_VENV_IN_PROJECT=1 pipenv lock
    PIPENV_VENV_IN_PROJECT=1 pipenv sync --dev

# docker host-mapped venv cannot be shared for localdev; container modified files not remapped to host user; pipenv sync is slow for subsequent cmds
virt SUBCOMMAND FORCE="noforce":
    #!/usr/bin/env bash
    if [ "{{FORCE}}" = "--force" ]  || [ "{{FORCE}}" = "-f" ]; then
        docker container prune --force
        docker volume rm --force {{NAME}}_pyvenv
    fi
    docker run -i -v `pwd`:`pwd` -v {{NAME}}_pyvenv:`pwd`/.venv -w `pwd` {{DEV_IMAGE}} just init {{SUBCOMMAND}}

@lint:
    pipenv run ruff check {{SRC_FOLDER}} {{TEST_FOLDER}}
    pipenv run ruff format --check {{SRC_FOLDER}} {{TEST_FOLDER}}

@typecheck:
    pipenv run mypy --explicit-package-bases -p {{NAME}}
    pipenv run mypy --allow-untyped-defs tests

@test:
    pipenv run pytest --hypothesis-show-statistics {{TEST_FOLDER}}

@format:
    pipenv run ruff check --fix-only {{SRC_FOLDER}} {{TEST_FOLDER}}
    pipenv run ruff format {{SRC_FOLDER}} {{TEST_FOLDER}}

@stats:
    pipenv run coverage run -m pytest {{TEST_FOLDER}}
    pipenv run coverage report -m
    scc --by-file --include-ext py

crossverify:
    #!/usr/bin/env bash
    set -euxo pipefail

    for py in 3.8.15 3.9.15 3.10.8 3.11.4
    do
        pyenv install -s $py
        pyenv local $py
        python -m venv /tmp/$py-crossverify
        source /tmp/$py-crossverify/bin/activate > /dev/null 2> /dev/null
        python --version
        pip -q install ruff mypy pytest hypothesis
        pip -q install -e .
        ruff check {{SRC_FOLDER}} {{TEST_FOLDER}}
        mypy --explicit-package-bases -p {{NAME}}
        mypy --allow-untyped-defs tests
        pytest --hypothesis-show-statistics {{TEST_FOLDER}}
        deactivate > /dev/null 2> /dev/null
        rm -rf /tmp/$py-crossverify
        pyenv local --unset
    done

######
## Custom Section Begin
######

######
## Custom Section End
######
