# laplace-sample

The optional custom-sampler runtime for laplace. It runs a compiled model
through [BridgeStan](https://github.com/roualdes/bridgestan) with one of the
built-in samplers (`nuts`, `rwm`, `mala`, `hmc`) and writes one CmdStan-style
CSV per chain, readable with `cmdstanr::as_cmdstan_fit()` (or
`cmdlaplacer::laplace_sample()` from R). CmdStan stays the default way to run
a laplace model; this is a second, separate binary.

```sh
cargo install --path laplace-sample --root ~/.local --force
laplace sample model.laplace --data data.json --sampler mala --seed 1
```

`laplace sample ...` finds `laplace-sample` on `PATH` and forwards its
arguments; `laplace-sample sample --help` lists every flag and each
sampler's `--set` settings.

## How a model gets built

- A `.laplace` file is compiled with `laplace build` first (in its own
  directory, so `build/<name>.stan` stays on disk); a `.stan` file is used
  as is.
- The BridgeStan source (pinned 2.9.0) is downloaded once to
  `~/.laplace/bridgestan/` and verified against a pinned SHA-256. Its
  Makefile then downloads the matching `stanc`. Point `LAPLACE_BRIDGESTAN` at
  an existing BridgeStan 2.9.0 tree to skip the download.
- The compiled model library is cached in `~/.laplace/cache/bridgestan/`
  (`LAPLACE_SAMPLE_CACHE` to move it), keyed by the `.stan` text, every file
  it `#include`s, the BridgeStan version and the toolchain below. A rerun of
  an unchanged model does not recompile.

### Which C++ compiler

The first match wins, and the choice is printed on every run
(`Compiler: ...`):

1. `LAPLACE_CXX`, then `CXX`, from the environment.
2. The `CXX` set in the `make/local` of the CmdStan that cmdstanr uses:
   `CMDSTAN` if set, else the newest `~/.cmdstan/cmdstan-<version>`. Only
   `CXX` is taken from that file.
3. Otherwise `make`'s default: BridgeStan's own `make/local` if it sets
   `CXX`, else the system compiler.

Step 2 is there so laplace-sample and CmdStan build models with the same
compiler, so their timings compare like with like. It is not a promise of
speed. The compiler changes gradient time a lot, and which one is faster
depends on the model. Measured with the same BridgeStan build
(`docs/sampler/phase-1-report.md`, "Follow-up"):

| Model | g++ | clang++ |
| --- | --- | --- |
| gp (`gps::rbf_cov`) | 203 µs | 148 µs |
| survival (Makeham) | 70 µs | 121 µs |
| ts (AR) | 21 µs | 26 µs |

For the fastest build of a particular model, try both with `LAPLACE_CXX`.

## Environment variables

| Variable | Effect |
| --- | --- |
| `LAPLACE_CXX` | C++ compiler for model builds (takes precedence over `CXX`) |
| `CXX` | C++ compiler, if `LAPLACE_CXX` is not set |
| `CMDSTAN` | CmdStan installation whose `make/local` `CXX` is used |
| `LAPLACE_BRIDGESTAN` | Use this BridgeStan source tree instead of downloading |
| `LAPLACE_SAMPLE_CACHE` | Model build cache directory |
| `LAPLACE_BIN` | The `laplace` binary used for `.laplace` inputs |
| `STANC` | Passed through to BridgeStan's Makefile: use this `stanc` |
