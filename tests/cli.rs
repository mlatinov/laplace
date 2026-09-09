//! End-to-end tests that run the actual compiled `laplace` binary against a
//! temporary project directory and a temporary fake "home" (so `add`,
//! `install`, and the registry never touch the real filesystem outside the
//! test's own tempdirs).

use std::fs;
use std::process::{Command, Output};

struct Project {
    _tmp: tempfile::TempDir,
    dir: std::path::PathBuf,
    home: std::path::PathBuf,
    extra_env: Vec<(String, String)>,
}

fn setup() -> Project {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("project");
    let home = tmp.path().join("home");
    fs::create_dir_all(&dir).unwrap();
    fs::create_dir_all(home.join(".laplace").join("registry")).unwrap();
    Project {
        dir,
        home,
        _tmp: tmp,
        extra_env: Vec::new(),
    }
}

impl Project {
    fn registry_dir(&self) -> std::path::PathBuf {
        self.home.join(".laplace").join("registry")
    }

    fn write_package(&self, name: &str, version: &str, exports: &[&str], stan_body: &str) {
        self.write_lib_package(name, version, exports, &[], stan_body, "stan");
    }

    /// A package with its own `[dependencies]`, written as a `.laplacelib`
    /// file so it can carry a `library { }` block.
    fn write_laplacelib_package(
        &self,
        name: &str,
        version: &str,
        exports: &[&str],
        deps: &[(&str, &str)],
        body: &str,
    ) {
        self.write_lib_package(name, version, exports, deps, body, "laplacelib");
    }

    fn write_lib_package(
        &self,
        name: &str,
        version: &str,
        exports: &[&str],
        deps: &[(&str, &str)],
        body: &str,
        extension: &str,
    ) {
        let pkg_dir = self.registry_dir().join(name).join(version);
        fs::create_dir_all(&pkg_dir).unwrap();
        let exports_toml = exports
            .iter()
            .map(|e| format!("\"{e}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let mut manifest =
            format!("name = \"{name}\"\nversion = \"{version}\"\nexports = [{exports_toml}]\n");
        if !deps.is_empty() {
            manifest.push_str("\n[dependencies]\n");
            for (dep, range) in deps {
                manifest.push_str(&format!("{dep} = \"{range}\"\n"));
            }
        }
        fs::write(pkg_dir.join("laplace.toml"), manifest).unwrap();
        fs::write(pkg_dir.join(format!("{name}.{extension}")), body).unwrap();
    }

    fn write_package_file(&self, name: &str, version: &str, file: &str, contents: &str) {
        let pkg_dir = self.registry_dir().join(name).join(version);
        fs::create_dir_all(&pkg_dir).unwrap();
        fs::write(pkg_dir.join(file), contents).unwrap();
    }

    fn project_file_exists(&self, name: &str) -> bool {
        self.dir.join(name).exists()
    }

    fn write_project_file(&self, name: &str, contents: &str) {
        fs::write(self.dir.join(name), contents).unwrap();
    }

    fn read_project_file(&self, name: &str) -> String {
        fs::read_to_string(self.dir.join(name)).unwrap()
    }

    fn run(&self, args: &[&str]) -> Output {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_laplace"));
        cmd.args(args)
            .current_dir(&self.dir)
            .env("HOME", &self.home)
            .env_remove("LAPLACE_REGISTRY");
        for (key, value) in &self.extra_env {
            cmd.env(key, value);
        }
        cmd.output().expect("failed to run laplace binary")
    }

    /// Write a fake `stanc` shell script that always exits with `exit_code`
    /// and prints `output` to stderr, and point `LAPLACE_STANC` at it, so
    /// `--validate` tests never depend on a real stanc install.
    fn fake_stanc(&mut self, exit_code: i32, output: &str) {
        let script = self.dir.join("fake_stanc.sh");
        fs::write(&script, format!("#!/bin/sh\ncat <<'EOF' 1>&2\n{output}\nEOF\nexit {exit_code}\n")).unwrap();
        let mut perms = fs::metadata(&script).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
        fs::set_permissions(&script, perms).unwrap();
        self.extra_env
            .push(("LAPLACE_STANC".to_string(), script.to_string_lossy().to_string()));
    }
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

const GPS_STAN: &str = r#"// @laplace
// @brief Squared exponential covariance matrix.
matrix rbf_cov(vector x, real alpha, real rho) {
  return gp_exp_quad_cov(x, alpha, rho);
}
"#;

const MODEL_LAPLACE: &str = r#"library {
  import gps
}

data {
  int<lower=1> N;
  vector[N] x;
}

parameters {
  real<lower=0> alpha;
  real<lower=0> rho;
}

model {
  matrix[N, N] K = gps::rbf_cov(x, alpha, rho);
  x ~ multi_normal(rep_vector(0, N), K);
}
"#;

#[test]
fn init_generates_manifest_from_documented_and_undocumented_functions() {
    let project = setup();
    project.write_project_file(
        "gps.stan",
        r#"// @laplace
// @brief Squared exponential covariance matrix.
matrix rbf_cov(vector x, real alpha, real rho) {
  return gp_exp_quad_cov(x, alpha, rho);
}

real jitter(real epsilon) {
  return epsilon;
}
"#,
    );

    let out = project.run(&["init"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("wrote laplace.toml"));
    assert!(text.contains("rbf_cov"));
    assert!(text.contains("jitter"));

    let manifest = project.read_project_file("laplace.toml");
    assert!(manifest.contains("version = \"0.1.0\""));
    assert!(manifest.contains("rbf_cov"));
    assert!(!manifest.contains("jitter"));
}

#[test]
fn init_does_not_overwrite_an_existing_manifest() {
    let project = setup();
    project.write_project_file("laplace.toml", "name = \"gps\"\nversion = \"9.9.9\"\nexports = []\n");
    project.write_project_file(
        "gps.stan",
        "real jitter(real epsilon) {\n  return epsilon;\n}\n",
    );

    let out = project.run(&["init"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("already exists"), "{}", stderr(&out));
    assert!(project.read_project_file("laplace.toml").contains("9.9.9"));
}

#[test]
fn build_without_installing_first_fails_with_instructions() {
    let project = setup();
    project.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);
    project.write_project_file("model.laplace", MODEL_LAPLACE);

    let out = project.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("laplace add"), "{}", stderr(&out));
}

#[test]
fn add_install_build_end_to_end() {
    let project = setup();
    project.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);

    let add_out = project.run(&["add", "gps"]);
    assert!(add_out.status.success(), "{}", stderr(&add_out));
    assert!(stdout(&add_out).contains("added gps@1.0.0"));

    let manifest = project.read_project_file("laplace.toml");
    assert!(manifest.contains("gps"));
    let lock = project.read_project_file("laplace.lock");
    assert!(lock.contains("1.0.0"));

    let install_out = project.run(&["install"]);
    assert!(install_out.status.success(), "{}", stderr(&install_out));
    assert!(stdout(&install_out).contains("installed 1 package"));
    assert!(project
        .home
        .join(".laplace/packages/gps/1.0.0/laplace.toml")
        .is_file());

    project.write_project_file("model.laplace", MODEL_LAPLACE);
    let build_out = project.run(&["build", "model.laplace"]);
    assert!(build_out.status.success(), "{}", stderr(&build_out));
    assert!(stdout(&build_out).contains("wrote build/model.stan"));
    assert!(stdout(&build_out).contains("1 dependency"));

    let compiled = project.read_project_file("build/model.stan");
    assert!(!compiled.contains("library"));
    assert!(!compiled.contains("gps::rbf_cov"));
    assert!(compiled.contains("gps__rbf_cov"));

    // --check against an up-to-date build succeeds without rewriting.
    let check_out = project.run(&["build", "model.laplace", "--check"]);
    assert!(check_out.status.success(), "{}", stderr(&check_out));
    assert!(stdout(&check_out).contains("up to date"));

    // Hand-edit the committed output to simulate drift; --check must catch it.
    fs::write(project.dir.join("build/model.stan"), "stale\n").unwrap();
    let stale_check = project.run(&["build", "model.laplace", "--check"]);
    assert!(!stale_check.status.success());
    assert!(stderr(&stale_check).contains("out of date"), "{}", stderr(&stale_check));

    // The stale file on disk must be untouched by --check.
    assert_eq!(project.read_project_file("build/model.stan"), "stale\n");
}

#[test]
fn build_output_is_byte_identical_across_runs() {
    let project = setup();
    project.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);
    project.run(&["add", "gps"]);
    project.run(&["install"]);
    project.write_project_file("model.laplace", MODEL_LAPLACE);

    project.run(&["build", "model.laplace", "-o", "out1.stan"]);
    project.run(&["build", "model.laplace", "-o", "out2.stan"]);

    assert_eq!(
        project.read_project_file("out1.stan"),
        project.read_project_file("out2.stan")
    );
}

#[test]
fn update_picks_up_a_newer_compatible_version() {
    let project = setup();
    project.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);
    let add_out = project.run(&["add", "gps"]);
    assert!(add_out.status.success(), "{}", stderr(&add_out));

    project.write_package("gps", "1.1.0", &["rbf_cov"], GPS_STAN);
    let update_out = project.run(&["update", "gps"]);
    assert!(update_out.status.success(), "{}", stderr(&update_out));
    assert!(stdout(&update_out).contains("updated gps to 1.1.0"));

    assert!(project.read_project_file("laplace.lock").contains("1.1.0"));
}

#[test]
fn add_unknown_package_fails_clearly() {
    let project = setup();
    let out = project.run(&["add", "nonexistent"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("nonexistent"), "{}", stderr(&out));
}

#[test]
fn calling_a_non_exported_function_fails_the_build() {
    let project = setup();
    project.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);
    project.run(&["add", "gps"]);
    project.run(&["install"]);

    project.write_project_file(
        "model.laplace",
        "library {\n  import gps\n}\nmodel {\n  real y = gps::secret(1.0);\n}\n",
    );

    let out = project.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("secret"), "{}", stderr(&out));
}

#[test]
fn build_with_no_library_block_is_a_pure_passthrough() {
    let project = setup();
    let source = "data {\n  int n;\n}\nmodel {\n}\n";
    project.write_project_file("model.laplace", source);

    let out = project.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(project.read_project_file("build/model.stan"), source);
    assert!(stdout(&out).contains("0 dependencies"));
}

#[test]
fn doc_renders_a_documented_function() {
    let project = setup();
    project.write_package(
        "gps",
        "1.0.0",
        &["rbf_cov"],
        r#"// @laplace
// @brief Squared exponential covariance matrix.
// @param x Vector of input locations.
// @return An N x N covariance matrix.
// @example rbf_cov(x, 1.0, 0.5)
matrix rbf_cov(vector x, real alpha, real rho) {
  return gp_exp_quad_cov(x, alpha, rho);
}
"#,
    );
    project.run(&["add", "gps"]);
    project.run(&["install"]);

    let out = project.run(&["doc", "gps::rbf_cov"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("gps::rbf_cov(x: vector, alpha: real, rho: real) -> matrix"));
    assert!(text.contains("Squared exponential covariance matrix."));
    assert!(text.contains("Parameters:"));
    assert!(text.contains("Returns:"));
    assert!(text.contains("Example:"));
}

#[test]
fn doc_html_renders_math_and_multiline_example() {
    let project = setup();
    project.write_package(
        "gps",
        "1.0.0",
        &["rbf_cov"],
        r#"// @laplace
// @brief Squared exponential covariance matrix.
// @math k(x, x') = \alpha^2 \exp\left(-\frac{(x - x')^2}{2 \rho^2}\right)
// @param x Vector of input locations.
// @return An N x N covariance matrix.
// @example matrix k = rbf_cov(x, 1.0, 0.5);
//   print(k);
matrix rbf_cov(vector x, real alpha, real rho) {
  return gp_exp_quad_cov(x, alpha, rho);
}
"#,
    );
    project.run(&["add", "gps"]);
    project.run(&["install"]);

    let out_path = project.dir.join("rbf_cov.html");
    let out = project.run(&[
        "doc",
        "gps::rbf_cov",
        "--html",
        "-o",
        out_path.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "{}", stderr(&out));

    let html = fs::read_to_string(&out_path).unwrap();
    assert!(html.contains("katex"));
    assert!(html.contains("class=\"math\""));
    assert!(html.contains("k(x, x&#39;)"));
    assert!(html.contains("<pre><code>matrix k = rbf_cov(x, 1.0, 0.5);\nprint(k);</code></pre>"));
}

#[test]
fn doc_html_without_math_tag_has_no_katex_span() {
    let project = setup();
    project.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);
    project.run(&["add", "gps"]);
    project.run(&["install"]);

    let out_path = project.dir.join("rbf_cov.html");
    let out = project.run(&[
        "doc",
        "gps::rbf_cov",
        "--html",
        "-o",
        out_path.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "{}", stderr(&out));

    let html = fs::read_to_string(&out_path).unwrap();
    assert!(!html.contains("class=\"math\""));
}

#[test]
fn doc_on_undocumented_function_notes_no_docs_available() {
    let project = setup();
    project.write_package(
        "gps",
        "1.0.0",
        &["jitter"],
        "real jitter(real epsilon) {\n  return epsilon;\n}\n",
    );
    project.run(&["add", "gps"]);
    project.run(&["install"]);

    let out = project.run(&["doc", "gps::jitter"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stdout(&out).contains("no @laplace documentation available"));
}

#[test]
fn doc_on_unknown_function_fails_clearly() {
    let project = setup();
    project.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);
    project.run(&["add", "gps"]);
    project.run(&["install"]);

    let out = project.run(&["doc", "gps::nonexistent"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("nonexistent"), "{}", stderr(&out));
}

#[test]
fn doc_on_package_never_added_fails_clearly() {
    let project = setup();
    let out = project.run(&["doc", "gps::rbf_cov"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("laplace.lock"), "{}", stderr(&out));
}

#[test]
fn doc_with_malformed_spec_fails_clearly() {
    let project = setup();
    let out = project.run(&["doc", "not-a-valid-spec"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("package"), "{}", stderr(&out));
}

#[test]
fn build_validate_passes_through_a_successful_stanc_run() {
    let mut project = setup();
    project.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);
    project.run(&["add", "gps"]);
    project.run(&["install"]);
    project.write_project_file("model.laplace", MODEL_LAPLACE);

    project.fake_stanc(0, "");
    let out = project.run(&["build", "model.laplace", "--validate"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stdout(&out).contains("stanc: OK"));
}

#[test]
fn build_validate_reports_stanc_errors_annotated_with_the_culprit_package() {
    let mut project = setup();
    project.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);
    project.run(&["add", "gps"]);
    project.run(&["install"]);
    project.write_project_file("model.laplace", MODEL_LAPLACE);

    // Build once (without --validate) to find out which output line the
    // gps-contributed function actually landed on, so the fake stanc error
    // can point at a real line inside that range.
    project.run(&["build", "model.laplace"]);
    let compiled = project.read_project_file("build/model.stan");
    let gps_line = compiled
        .lines()
        .position(|l| l.contains("gps__rbf_cov"))
        .unwrap()
        + 1;

    project.fake_stanc(1, &format!("Semantic error at 'model.stan', line {gps_line}, column 1"));
    let out = project.run(&["build", "model.laplace", "--validate"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("came from package `gps`"), "{err}");
    assert!(err.contains("Semantic error"), "{err}");
}

#[test]
fn build_validate_without_stanc_installed_fails_clearly() {
    let mut project = setup();
    let source = "data {\n  int n;\n}\nmodel {\n}\n";
    project.write_project_file("model.laplace", source);
    project
        .extra_env
        .push(("LAPLACE_STANC".to_string(), "laplace-test-no-such-command".to_string()));

    let out = project.run(&["build", "model.laplace", "--validate"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("not found"), "{}", stderr(&out));
    // The build itself must still have written its output before validation ran.
    assert_eq!(project.read_project_file("build/model.stan"), source);
}

#[test]
fn build_check_never_invokes_validate() {
    let mut project = setup();
    let source = "data {\n  int n;\n}\nmodel {\n}\n";
    project.write_project_file("model.laplace", source);
    project.run(&["build", "model.laplace"]);

    // If --validate ran here it would fail (bogus command); --check must
    // short-circuit before ever reaching it.
    project
        .extra_env
        .push(("LAPLACE_STANC".to_string(), "laplace-test-no-such-command".to_string()));
    let out = project.run(&["build", "model.laplace", "--check", "--validate"]);
    assert!(out.status.success(), "{}", stderr(&out));
}

// ---------------------------------------------------------------------------
// Feature A: `--split-functions`
// ---------------------------------------------------------------------------

#[test]
fn split_functions_emits_an_include_and_a_stanfunctions_file() {
    let p = setup();
    p.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);
    p.write_project_file("model.laplace", MODEL_LAPLACE);

    assert!(p.run(&["add", "gps"]).status.success());
    assert!(p.run(&["install"]).status.success());

    let out = p.run(&["build", "model.laplace", "--split-functions"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stdout(&out).contains("gps.stanfunctions"), "{}", stdout(&out));
    assert!(stdout(&out).contains("--include-paths"), "{}", stdout(&out));

    let stan = p.read_project_file("build/model.stan");
    assert!(stan.contains("#include \"gps.stanfunctions\""), "{stan}");
    assert!(!stan.contains("gp_exp_quad_cov"), "{stan}");
    assert!(stan.contains("gps__rbf_cov(x, alpha, rho)"), "{stan}");

    // The functions file sits next to the .stan file, where stanc looks first.
    let funcs = p.read_project_file("build/gps.stanfunctions");
    assert!(funcs.contains("matrix gps__rbf_cov(vector x"), "{funcs}");
    assert!(!funcs.contains("functions {"), "{funcs}");
}

#[test]
fn split_functions_output_is_byte_identical_across_runs() {
    let p = setup();
    p.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);
    p.write_project_file("model.laplace", MODEL_LAPLACE);
    assert!(p.run(&["add", "gps"]).status.success());
    assert!(p.run(&["install"]).status.success());

    assert!(p
        .run(&["build", "model.laplace", "--split-functions"])
        .status
        .success());
    let first_stan = p.read_project_file("build/model.stan");
    let first_funcs = p.read_project_file("build/gps.stanfunctions");

    assert!(p
        .run(&["build", "model.laplace", "--split-functions"])
        .status
        .success());
    assert_eq!(first_stan, p.read_project_file("build/model.stan"));
    assert_eq!(first_funcs, p.read_project_file("build/gps.stanfunctions"));
}

#[test]
fn split_functions_is_a_no_op_for_a_project_with_no_imports() {
    let p = setup();
    let source = "data {\n  int<lower=1> N;\n}\nmodel {\n}\n";
    p.write_project_file("model.laplace", source);

    assert!(p.run(&["build", "model.laplace"]).status.success());
    let plain = p.read_project_file("build/model.stan");

    assert!(p
        .run(&["build", "model.laplace", "--split-functions"])
        .status
        .success());
    assert_eq!(p.read_project_file("build/model.stan"), plain);
    assert_eq!(plain, source);
    assert!(!p.project_file_exists("build/model.stanfunctions"));
}

#[test]
fn a_multi_file_package_concatenates_into_one_stanfunctions_file() {
    let p = setup();
    p.write_package("multi", "1.0.0", &["one", "two"], "real one() {\n  return 1;\n}\n");
    p.write_package_file(
        "multi",
        "1.0.0",
        "zz_extra.stan",
        "real two() {\n  return 2;\n}\n",
    );
    p.write_project_file(
        "model.laplace",
        "library {\n  import multi\n}\nmodel {\n  real y = multi::one() + multi::two();\n}\n",
    );

    assert!(p.run(&["add", "multi"]).status.success());
    assert!(p.run(&["install"]).status.success());
    let out = p.run(&["build", "model.laplace", "--split-functions"]);
    assert!(out.status.success(), "{}", stderr(&out));

    let funcs = p.read_project_file("build/multi.stanfunctions");
    assert!(funcs.contains("real multi__one()"), "{funcs}");
    assert!(funcs.contains("real multi__two()"), "{funcs}");
    assert!(!p.project_file_exists("build/zz_extra.stanfunctions"));
}

#[test]
fn split_functions_check_flags_a_stale_stanfunctions_file() {
    let p = setup();
    p.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);
    p.write_project_file("model.laplace", MODEL_LAPLACE);
    assert!(p.run(&["add", "gps"]).status.success());
    assert!(p.run(&["install"]).status.success());
    assert!(p
        .run(&["build", "model.laplace", "--split-functions"])
        .status
        .success());

    assert!(p
        .run(&["build", "model.laplace", "--split-functions", "--check"])
        .status
        .success());

    p.write_project_file("build/gps.stanfunctions", "// tampered\n");
    let out = p.run(&["build", "model.laplace", "--split-functions", "--check"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("gps.stanfunctions"), "{}", stderr(&out));
}

// ---------------------------------------------------------------------------
// Feature B: `.laplacelib` libraries that depend on libraries
// ---------------------------------------------------------------------------

const STATS_STAN: &str = r#"// @laplace
// @brief Arithmetic mean of a vector.
// @param x The vector to average.
// @return The mean of `x`.
real mean_(vector x) {
  return sum(x) / num_elements(x);
}
"#;

const REGRESSION_LIB: &str = r#"library {
  import stats
}

// @laplace
// @brief Centre a vector on its mean.
// @param x The vector to centre.
// @return `x` minus its mean.
vector centre(vector x) {
  return x - stats::mean_(x);
}
"#;

const CHAIN_MODEL: &str = r#"library {
  import regression
}

data {
  int<lower=1> N;
  vector[N] y;
}

model {
  vector[N] c = regression::centre(y);
  c ~ std_normal();
}
"#;

/// `stats` (leaf, plain .stan) <- `regression` (.laplacelib importing stats).
fn write_chain(p: &Project, stats_version: &str, regression_range: &str) {
    p.write_package("stats", stats_version, &["mean_"], STATS_STAN);
    p.write_laplacelib_package(
        "regression",
        "1.0.0",
        &["centre"],
        &[("stats", regression_range)],
        REGRESSION_LIB,
    );
}

#[test]
fn adding_a_library_pulls_in_its_own_dependencies() {
    let p = setup();
    write_chain(&p, "1.0.0", "^1.0");
    p.write_project_file("model.laplace", CHAIN_MODEL);

    let out = p.run(&["add", "regression"]);
    assert!(out.status.success(), "{}", stderr(&out));

    // laplace.toml records only what the user asked for...
    let toml = p.read_project_file("laplace.toml");
    assert!(toml.contains("regression"), "{toml}");
    assert!(!toml.contains("stats"), "{toml}");

    // ...while the lock records the whole graph, with the edge between them.
    let lock = p.read_project_file("laplace.lock");
    assert!(lock.contains("root = [\"regression\"]"), "{lock}");
    assert!(lock.contains("name = \"stats\""), "{lock}");
    assert!(lock.contains("dependencies = [\"stats\"]"), "{lock}");
}

#[test]
fn a_three_package_chain_builds_with_every_level_mangled() {
    let p = setup();
    write_chain(&p, "1.0.0", "^1.0");
    p.write_project_file("model.laplace", CHAIN_MODEL);

    assert!(p.run(&["add", "regression"]).status.success());
    assert!(p.run(&["install"]).status.success());
    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stdout(&out).contains("1 transitive"), "{}", stdout(&out));

    let stan = p.read_project_file("build/model.stan");
    // The transitive package is present and callable from its dependent.
    assert!(stan.contains("real stats__mean_(vector x)"), "{stan}");
    assert!(stan.contains("vector regression__centre(vector x)"), "{stan}");
    assert!(stan.contains("return x - stats__mean_(x);"), "{stan}");
    assert!(stan.contains("regression__centre(y)"), "{stan}");
    // No unresolved `::` and no leftover `library { }` block survive.
    assert!(!stan.contains("::"), "{stan}");
    assert!(!stan.contains("library {"), "{stan}");
    // Defined before used.
    assert!(stan.find("stats__mean_").unwrap() < stan.find("regression__centre").unwrap());
    // Each symbol is defined exactly once.
    assert_eq!(stan.matches("real stats__mean_(vector x)").count(), 1);
}

#[test]
fn a_transitive_dependency_is_not_callable_from_the_project() {
    let p = setup();
    write_chain(&p, "1.0.0", "^1.0");
    p.write_project_file(
        "model.laplace",
        "library {\n  import regression\n}\nmodel {\n  real m = stats::mean_([1.0]');\n}\n",
    );

    assert!(p.run(&["add", "regression"]).status.success());
    assert!(p.run(&["install"]).status.success());

    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("stats::mean_"), "{err}");
    assert!(err.contains("library"), "{err}");
}

#[test]
fn importing_a_transitive_only_package_directly_tells_you_to_add_it() {
    let p = setup();
    write_chain(&p, "1.0.0", "^1.0");
    assert!(p.run(&["add", "regression"]).status.success());
    assert!(p.run(&["install"]).status.success());

    // `stats` is installed (as regression's dependency) but is not a direct
    // dependency of the project, so importing it must not silently work.
    p.write_project_file(
        "model.laplace",
        "library {\n  import stats\n}\nmodel {\n}\n",
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("transitive"), "{err}");
    assert!(err.contains("laplace add stats"), "{err}");
}

#[test]
fn a_diamond_with_compatible_ranges_resolves_to_one_shared_package() {
    let p = setup();
    p.write_package("stats", "1.0.0", &["mean_"], STATS_STAN);
    p.write_package("stats", "1.4.0", &["mean_"], STATS_STAN);
    p.write_laplacelib_package(
        "regression",
        "1.0.0",
        &["centre"],
        &[("stats", ">=1.1, <2.0")],
        REGRESSION_LIB,
    );
    p.write_project_file(
        "model.laplace",
        r#"library {
  import regression
  import stats
}

data {
  int<lower=1> N;
  vector[N] y;
}

model {
  vector[N] c = regression::centre(y);
  c ~ normal(stats::mean_(y), 1);
}
"#,
    );

    assert!(p.run(&["add", "regression"]).status.success());
    assert!(p.run(&["add", "stats"]).status.success());
    assert!(p.run(&["install"]).status.success());
    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));

    // One `stats`, satisfying both `^1.4.0` (written by add) and `>=1.1, <2.0`.
    let lock = p.read_project_file("laplace.lock");
    assert_eq!(lock.matches("name = \"stats\"").count(), 1, "{lock}");
    assert!(lock.contains("version = \"1.4.0\""), "{lock}");

    let stan = p.read_project_file("build/model.stan");
    assert_eq!(
        stan.matches("real stats__mean_(vector x)").count(),
        1,
        "the shared package must not be duplicated:\n{stan}"
    );
}

#[test]
fn a_diamond_with_incompatible_ranges_names_both_requirers() {
    let p = setup();
    p.write_package("stats", "1.0.0", &["mean_"], STATS_STAN);
    p.write_package("stats", "2.0.0", &["mean_"], STATS_STAN);
    p.write_laplacelib_package(
        "regression",
        "1.0.0",
        &["centre"],
        &[("stats", "^2.0")],
        REGRESSION_LIB,
    );

    // Pin the project to stats ^1.0, which regression's ^2.0 contradicts.
    p.write_project_file(
        "laplace.toml",
        "[dependencies]\nstats = \"^1.0\"\n",
    );

    let out = p.run(&["add", "regression"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("stats"), "{err}");
    assert!(err.contains("this project"), "{err}");
    assert!(err.contains("regression@1.0.0"), "{err}");
    assert!(err.contains("^1"), "{err}");
    assert!(err.contains("^2"), "{err}");
}

#[test]
fn a_dependency_cycle_is_a_hard_error_with_the_path_printed() {
    let p = setup();
    p.write_laplacelib_package(
        "alpha",
        "1.0.0",
        &["a"],
        &[("beta", "^1.0")],
        "library {\n  import beta\n}\nreal a() {\n  return beta::b();\n}\n",
    );
    p.write_laplacelib_package(
        "beta",
        "1.0.0",
        &["b"],
        &[("alpha", "^1.0")],
        "library {\n  import alpha\n}\nreal b() {\n  return alpha::a();\n}\n",
    );

    let out = p.run(&["add", "alpha"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("dependency cycle"), "{err}");
    assert!(err.contains("alpha -> beta -> alpha"), "{err}");
}

#[test]
fn install_reconstructs_the_whole_graph_from_the_lock_alone() {
    let p = setup();
    write_chain(&p, "1.0.0", "^1.0");
    p.write_project_file("model.laplace", CHAIN_MODEL);

    assert!(p.run(&["add", "regression"]).status.success());
    assert!(p.run(&["install"]).status.success());
    assert!(p.run(&["build", "model.laplace"]).status.success());
    let expected = p.read_project_file("build/model.stan");
    let lock_before = p.read_project_file("laplace.lock");

    // Fresh machine: blow away the install cache and the manifest's ranges,
    // leaving only the lock. `install` must not re-resolve anything.
    fs::remove_dir_all(p.home.join(".laplace").join("packages")).unwrap();
    fs::write(p.dir.join("laplace.toml"), "[dependencies]\n").unwrap();

    let out = p.run(&["install"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stdout(&out).contains("2 packages"), "{}", stdout(&out));

    assert!(p.run(&["build", "model.laplace"]).status.success());
    assert_eq!(p.read_project_file("build/model.stan"), expected);
    assert_eq!(p.read_project_file("laplace.lock"), lock_before);
}

#[test]
fn a_transitive_dependency_is_flattened_into_its_parents_stanfunctions_file() {
    let p = setup();
    write_chain(&p, "1.0.0", "^1.0");
    p.write_project_file("model.laplace", CHAIN_MODEL);

    assert!(p.run(&["add", "regression"]).status.success());
    assert!(p.run(&["install"]).status.success());
    let out = p.run(&["build", "model.laplace", "--split-functions"]);
    assert!(out.status.success(), "{}", stderr(&out));

    let stan = p.read_project_file("build/model.stan");
    assert!(stan.contains("#include \"regression.stanfunctions\""), "{stan}");
    // `stats` is private to `regression`, so it gets no file of its own.
    assert!(!stan.contains("stats.stanfunctions"), "{stan}");
    assert!(!p.project_file_exists("build/stats.stanfunctions"));

    let funcs = p.read_project_file("build/regression.stanfunctions");
    assert!(funcs.contains("real stats__mean_(vector x)"), "{funcs}");
    assert!(funcs.contains("vector regression__centre(vector x)"), "{funcs}");
    assert!(funcs.contains("Bundled dependencies of `regression`: stats 1.0.0"), "{funcs}");
    assert!(funcs.find("stats__mean_").unwrap() < funcs.find("regression__centre").unwrap());
    assert!(!funcs.contains("functions {"), "{funcs}");
}

#[test]
fn a_diamond_in_split_mode_puts_the_shared_package_in_exactly_one_file() {
    let p = setup();
    write_chain(&p, "1.0.0", "^1.0");
    p.write_project_file(
        "model.laplace",
        r#"library {
  import regression
  import stats
}

data {
  int<lower=1> N;
  vector[N] y;
}

model {
  vector[N] c = regression::centre(y);
  c ~ normal(stats::mean_(y), 1);
}
"#,
    );

    assert!(p.run(&["add", "regression"]).status.success());
    assert!(p.run(&["add", "stats"]).status.success());
    assert!(p.run(&["install"]).status.success());
    assert!(p
        .run(&["build", "model.laplace", "--split-functions"])
        .status
        .success());

    let stats_funcs = p.read_project_file("build/stats.stanfunctions");
    let reg_funcs = p.read_project_file("build/regression.stanfunctions");
    assert!(stats_funcs.contains("real stats__mean_(vector x)"), "{stats_funcs}");
    assert!(
        !reg_funcs.contains("real stats__mean_(vector x)"),
        "the shared package must not be duplicated:\n{reg_funcs}"
    );

    // `stats` is included first, so its definitions precede regression's use.
    let stan = p.read_project_file("build/model.stan");
    assert!(
        stan.find("#include \"stats.").unwrap() < stan.find("#include \"regression.").unwrap(),
        "{stan}"
    );
}

#[test]
fn a_laplacelib_containing_a_model_block_fails_with_a_clear_error() {
    let p = setup();
    p.write_laplacelib_package(
        "bad",
        "1.0.0",
        &["f"],
        &[],
        "real f() {\n  return 1;\n}\n\nmodel {\n  y ~ normal(0, 1);\n}\n",
    );

    // The dialect check runs as soon as laplace reads the package's
    // sources, which is at `add` time -- long before anyone builds a model
    // against it.
    let out = p.run(&["add", "bad"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("bad.laplacelib"), "{err}");
    assert!(err.contains("`model` block"), "{err}");
    assert!(err.contains("not a model itself"), "{err}");
}

#[test]
fn a_laplacelib_importing_something_its_manifest_omits_fails_clearly() {
    let p = setup();
    p.write_package("stats", "1.0.0", &["mean_"], STATS_STAN);
    // No `[dependencies]` in the manifest, but the source imports `stats`.
    p.write_laplacelib_package("sneaky", "1.0.0", &["c"], &[], REGRESSION_LIB.replace("centre", "c").as_str());
    p.write_project_file(
        "model.laplace",
        "library {\n  import sneaky\n}\nmodel {\n}\n",
    );

    assert!(p.run(&["add", "sneaky"]).status.success());
    assert!(p.run(&["install"]).status.success());

    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("stats"), "{err}");
    assert!(err.contains("[dependencies]"), "{err}");
}

#[test]
fn a_deep_four_level_chain_builds_in_dependency_order() {
    let p = setup();
    p.write_package("base", "1.0.0", &["b"], "real b() {\n  return 1;\n}\n");
    p.write_laplacelib_package(
        "mid",
        "1.0.0",
        &["m"],
        &[("base", "^1.0")],
        "library {\n  import base\n}\nreal m() {\n  return base::b() + 1;\n}\n",
    );
    p.write_laplacelib_package(
        "top",
        "1.0.0",
        &["t"],
        &[("mid", "^1.0")],
        "library {\n  import mid\n}\nreal t() {\n  return mid::m() + 1;\n}\n",
    );
    p.write_project_file(
        "model.laplace",
        "library {\n  import top\n}\nmodel {\n  real y = top::t();\n}\n",
    );

    assert!(p.run(&["add", "top"]).status.success());
    assert!(p.run(&["install"]).status.success());
    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));

    let stan = p.read_project_file("build/model.stan");
    let base_at = stan.find("real base__b()").unwrap();
    let mid_at = stan.find("real mid__m()").unwrap();
    let top_at = stan.find("real top__t()").unwrap();
    assert!(base_at < mid_at && mid_at < top_at, "{stan}");
    assert!(stan.contains("return base__b() + 1;"), "{stan}");
    assert!(stan.contains("return mid__m() + 1;"), "{stan}");
    assert!(stan.contains("top__t()"), "{stan}");
}

#[test]
fn doc_lookup_works_for_a_laplacelib_package() {
    let p = setup();
    write_chain(&p, "1.0.0", "^1.0");
    assert!(p.run(&["add", "regression"]).status.success());
    assert!(p.run(&["install"]).status.success());

    let out = p.run(&["doc", "regression::centre"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("Centre a vector on its mean."), "{text}");
    assert!(text.contains("regression::centre"), "{text}");
}

/// The real acceptance test for `--split-functions`: hand the generated
/// output to a real `stanc` and confirm the `#include` resolves. Skipped
/// (rather than failed) where stanc isn't installed, since laplace itself
/// never requires it.
#[test]
fn stanc_accepts_the_split_output_when_stanc_is_available() {
    let Some(stanc) = real_stanc() else {
        eprintln!("skipping: no stanc on PATH (set LAPLACE_TEST_STANC to point at one)");
        return;
    };

    let p = setup();
    write_chain(&p, "1.0.0", "^1.0");
    p.write_project_file("model.laplace", CHAIN_MODEL);
    assert!(p.run(&["add", "regression"]).status.success());
    assert!(p.run(&["install"]).status.success());
    assert!(p
        .run(&["build", "model.laplace", "--split-functions"])
        .status
        .success());

    // The .stan file, whose #include must resolve relative to its own dir.
    let out = Command::new(&stanc)
        .arg(p.dir.join("build").join("model.stan"))
        .output()
        .expect("failed to run stanc");
    assert!(
        out.status.success(),
        "stanc rejected the split output:\n{}\n{}",
        stdout(&out),
        stderr(&out)
    );

    // ...and the .stanfunctions file must stand on its own as a
    // functions-only Stan file.
    let out = Command::new(&stanc)
        .arg(p.dir.join("build").join("regression.stanfunctions"))
        .output()
        .expect("failed to run stanc");
    assert!(
        out.status.success(),
        "stanc rejected the .stanfunctions file:\n{}\n{}",
        stdout(&out),
        stderr(&out)
    );
}

fn real_stanc() -> Option<String> {
    if let Ok(path) = std::env::var("LAPLACE_TEST_STANC") {
        return Some(path);
    }
    Command::new("stanc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| "stanc".to_string())
}

#[test]
fn a_missing_transitive_dependency_names_the_package_and_the_registry() {
    let p = setup();
    // `regression` declares a dependency on `stats`, but `stats` was never
    // published -- the user never wrote that name down, so the error has to.
    p.write_laplacelib_package(
        "regression",
        "1.0.0",
        &["centre"],
        &[("stats", "^1.0")],
        REGRESSION_LIB,
    );

    let out = p.run(&["add", "regression"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("stats"), "{err}");
    assert!(err.contains("registry"), "{err}");
}

#[test]
fn update_moves_only_the_named_package() {
    let p = setup();
    p.write_package("stats", "1.0.0", &["mean_"], STATS_STAN);
    p.write_package("other", "1.0.0", &["o"], "real o() {\n  return 1;\n}\n");

    assert!(p.run(&["add", "stats"]).status.success());
    assert!(p.run(&["add", "other"]).status.success());

    // Newer compatible versions of both land in the registry.
    p.write_package("stats", "1.5.0", &["mean_"], STATS_STAN);
    p.write_package("other", "1.5.0", &["o"], "real o() {\n  return 1;\n}\n");

    let out = p.run(&["update", "stats"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stdout(&out).contains("updated stats to 1.5.0"), "{}", stdout(&out));

    let lock = p.read_project_file("laplace.lock");
    assert!(lock.contains("version = \"1.5.0\""), "{lock}");
    assert!(
        lock.contains("version = \"1.0.0\""),
        "`other` must stay pinned at 1.0.0:\n{lock}"
    );
}
