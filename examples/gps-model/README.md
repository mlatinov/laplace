# gps-model

The end-to-end example referenced by `laplace-project-plan.md`: a Gaussian
process model whose covariance function comes from an imported `gps` package.

- `model.laplace` — the source, with a `library { }` block and one
  `gps::rbf_cov(...)` call site.
- `laplace.toml` — the hand-edited range (`gps = "^1.0.0"`).
- `laplace.lock` — the machine-written pin. `root` names the project's direct
  dependencies and each `[[package]]` carries its own `dependencies`, so the
  whole graph round-trips through `laplace install` without re-resolving.
  (`gps` is a leaf, so its list is empty.)
- `build/model.stan` — the compiled output, committed on purpose: it's meant to
  be readable and runnable without laplace installed.

To rebuild it you need `gps@1.0.0` in a registry laplace can see
(`~/.laplace/registry/gps/1.0.0/`, or point `LAPLACE_REGISTRY` elsewhere), then:

```sh
laplace install
laplace build model.laplace
```

Add `--split-functions` to see the other output shape: a `gps.stanfunctions`
file next to `build/model.stan`, `#include`d from its `functions { }` block.
