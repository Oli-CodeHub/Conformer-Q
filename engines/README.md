# Optional Local Tools

This directory is an optional local installation location. Third-party binaries
are intentionally not committed to the repository. Install them separately and
follow their own license terms.

## CREST

Place a compatible Apple Silicon macOS installation in `crest-3.0.2/`, or use a
system installation available as `crest` and `xtb` on `PATH`:

- CREST 3.0.2
- xTB 6.7.1

The app invokes CREST through an automatically created no-space symlink because
CREST cannot launch an external xTB binary from the current project path, which
contains spaces. On this platform CREST is run with the external xTB backend:

```text
--legacy --gfn2 -xnam <xtb>
```

This avoids the failing internal tblite initial-optimization path observed for
the EBF test molecule on this machine.

You can also set `MDW_CREST_BIN` and `MDW_XTB_BIN` to the executable paths.

## ORCA

The refinement workflow has been validated with ORCA 6.0.1. Obtain ORCA from
its official distribution channel and place it in `orca-6.0.1/`, with its
`orca` executable directly inside that directory, or set `MDW_ORCA_DIR` /
`MDW_ORCA_BIN`. ORCA is not distributed by this repository.

## Ketcher

Place the standalone distribution in
`ketcher-standalone-3.7.0/standalone/`, with `index.html` directly inside that
directory, or set `MDW_KETCHER_DIR`.
