# tools

## `pubcheck.sh`

Runs [ACL pubcheck](https://github.com/acl-org/aclpubcheck) against the paper.

```sh
tools/pubcheck.sh          # format checks only
tools/pubcheck.sh --refs   # also the bibliography check
```

Exit status is 0 when pubcheck reports zero **errors**. Warnings are printed but
do not fail the run.

### Why it builds a second copy of the paper

pubcheck is written for the **camera-ready** build. Our submission build uses
`\usepackage[review]{acl}`, which prints line numbers in the margins, and the
margin checker flags every one of them: running it directly on `acl_latex.pdf`
produces hundreds of spurious `Margin.LEFT` / `Margin.RIGHT` violations and tells
you nothing. The script therefore copies the source, swaps `[review]{acl}` for
`{acl}`, builds that, and checks the result. `paper/acl_latex.pdf` is never
touched.

Because that copy is not what gets submitted, the script also asserts three
things about the real source, which pubcheck cannot see:

- it is still in review mode,
- the author block is still anonymous,
- the build has no undefined references or citations.

### Run it locally

There is no CI for this: the paper needs XeLaTeX, the TikZ stack and the Bengali
font, and rebuilding all of that on a runner to re-check an artifact we already
build locally is not worth the maintenance. Run the script before you submit.

### Installation note

aclpubcheck and its `tsv` dependency use a `setup.py` that newer setuptools
rejects (`AttributeError: install_layout`). The script creates its own virtualenv
with `setuptools<60` rather than touching the ambient interpreter. Set
`PUBCHECK_VENV` to reuse one, and `PUBCHECK_WORKDIR` to keep the build
directory.

## `aclpubcheck_online.py`

The upstream Colab notebook, exported, kept for reference. It is the same tool
driven interactively; `pubcheck.sh` is the local wrapper.
