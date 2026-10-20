# Contributing

Bug reports, ideas and pull requests are welcome.

## Reporting a problem

Open an [issue](https://github.com/rzvky/Duplicate-Media-Finder/issues) and include the output of
`python -m dupefinder check`. If two files were matched wrongly (or a copy was missed), say where the
files came from (phone, WhatsApp, camera, export...) and, if you can, attach them.

## Working on the code

```bash
git clone https://github.com/rzvky/Duplicate-Media-Finder.git
cd Duplicate-Media-Finder
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt      # Windows: .venv\Scripts\python
.venv/bin/python -m pytest -q tests
```

The tests take a few minutes. Tests that need the AI model are skipped until the model is set up
(see the README), and video tests are skipped without the `av` package.

Before sending a pull request:

- run `ruff check .` and `ruff format .` (install with `pip install ruff==0.16.8`)
- add a test for the bug or feature in `tests/test_dupefinder.py`
- if you change how files are matched, run the accuracy check in [docs/accuracy.md](docs/accuracy.md)
  and mention the result in the pull request

The rule that matters most: **a different photo or video must never be ticked as a copy.**
Missing a duplicate is acceptable; moving a unique file is not.
