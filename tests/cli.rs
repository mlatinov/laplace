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
    ///
    /// `exports` names the package's public API. In a `.laplacelib` file
    /// that is spelled with the `pub` keyword rather than in the
    /// manifest, so the names are marked up in the body and the
    /// manifest's `exports` list is left empty.
    fn write_laplacelib_package(
        &self,
        name: &str,
        version: &str,
        exports: &[&str],
        deps: &[(&str, &str)],
        body: &str,
    ) {
        let body = mark_pub(body, exports);
        self.write_lib_package(name, version, &[], deps, &body, "laplacelib");
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
        fs::write(
            &script,
            format!("#!/bin/sh\ncat <<'EOF' 1>&2\n{output}\nEOF\nexit {exit_code}\n"),
        )
        .unwrap();
        let mut perms = fs::metadata(&script).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
        fs::set_permissions(&script, perms).unwrap();
        self.extra_env.push((
            "LAPLACE_STANC".to_string(),
            script.to_string_lossy().to_string(),
        ));
    }
}

/// Put `pub` in front of each named function's definition.
///
/// A definition is a line that declares `<type> <name>(`; a call like
/// `return pkg::name(x)` has no space before the name and is left alone.
fn mark_pub(body: &str, names: &[&str]) -> String {
    let mut out = String::with_capacity(body.len());
    for line in body.split_inclusive('\n') {
        let trimmed = line.trim_start();
        let is_definition = !trimmed.starts_with("//")
            && !trimmed.starts_with("return")
            && names.iter().any(|n| line.contains(&format!(" {n}(")));
        if is_definition {
            let indent = &line[..line.len() - trimmed.len()];
            out.push_str(indent);
            out.push_str("pub ");
            out.push_str(trimmed);
        } else {
            out.push_str(line);
        }
    }
    out
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
    project.write_project_file(
        "laplace.toml",
        "name = \"gps\"\nversion = \"9.9.9\"\nexports = []\n",
    );
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
    assert!(
        stderr(&stale_check).contains("out of date"),
        "{}",
        stderr(&stale_check)
    );

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

    project.fake_stanc(
        1,
        &format!("Semantic error at 'model.stan', line {gps_line}, column 1"),
    );
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
    project.extra_env.push((
        "LAPLACE_STANC".to_string(),
        "laplace-test-no-such-command".to_string(),
    ));

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
    project.extra_env.push((
        "LAPLACE_STANC".to_string(),
        "laplace-test-no-such-command".to_string(),
    ));
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
    assert!(
        stdout(&out).contains("gps.stanfunctions"),
        "{}",
        stdout(&out)
    );
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
    p.write_package(
        "multi",
        "1.0.0",
        &["one", "two"],
        "real one() {\n  return 1;\n}\n",
    );
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
    assert!(
        stderr(&out).contains("gps.stanfunctions"),
        "{}",
        stderr(&out)
    );
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
    assert!(
        stan.contains("vector regression__centre(vector x)"),
        "{stan}"
    );
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
    p.write_project_file("laplace.toml", "[dependencies]\nstats = \"^1.0\"\n");

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
    assert!(
        stan.contains("#include \"regression.stanfunctions\""),
        "{stan}"
    );
    // `stats` is private to `regression`, so it gets no file of its own.
    assert!(!stan.contains("stats.stanfunctions"), "{stan}");
    assert!(!p.project_file_exists("build/stats.stanfunctions"));

    let funcs = p.read_project_file("build/regression.stanfunctions");
    assert!(funcs.contains("real stats__mean_(vector x)"), "{funcs}");
    assert!(
        funcs.contains("vector regression__centre(vector x)"),
        "{funcs}"
    );
    assert!(
        funcs.contains("Bundled dependencies of `regression`: stats 1.0.0"),
        "{funcs}"
    );
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
    assert!(
        stats_funcs.contains("real stats__mean_(vector x)"),
        "{stats_funcs}"
    );
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
    p.write_laplacelib_package(
        "sneaky",
        "1.0.0",
        &["c"],
        &[],
        REGRESSION_LIB.replace("centre", "c").as_str(),
    );
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
    assert!(
        stdout(&out).contains("updated stats to 1.5.0"),
        "{}",
        stdout(&out)
    );

    let lock = p.read_project_file("laplace.lock");
    assert!(lock.contains("version = \"1.5.0\""), "{lock}");
    assert!(
        lock.contains("version = \"1.0.0\""),
        "`other` must stay pinned at 1.0.0:\n{lock}"
    );
}

// ---------------------------------------------------------------------------
// Ecosystem-test fixes
// ---------------------------------------------------------------------------

#[test]
fn build_validate_passes_the_include_path_and_keeps_stancs_cpp_out_of_build() {
    let mut project = setup();
    project.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);
    project.run(&["add", "gps"]);
    project.write_project_file("model.laplace", MODEL_LAPLACE);

    // A fake stanc that records its arguments, then writes to whatever
    // `--o=` names, the way the real one writes the .hpp.
    let args_file = project.dir.join("stanc_args.txt");
    let script = project.dir.join("recording_stanc.sh");
    fs::write(
        &script,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nfor a in \"$@\"; do case \"$a\" in --o=*) echo cpp > \"${{a#--o=}}\";; esac; done\nexit 0\n",
            args_file.display()
        ),
    )
    .unwrap();
    let mut perms = fs::metadata(&script).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
    fs::set_permissions(&script, perms).unwrap();
    project.extra_env.push((
        "LAPLACE_STANC".to_string(),
        script.to_string_lossy().to_string(),
    ));

    let out = project.run(&["build", "model.laplace", "--split-functions", "--validate"]);
    assert!(out.status.success(), "{}", stderr(&out));

    let args = fs::read_to_string(&args_file).unwrap();
    assert!(args.lines().any(|a| a == "--include-paths=build"), "{args}");
    assert!(args.lines().any(|a| a.starts_with("--o=")), "{args}");
    assert!(args.lines().any(|a| a == "build/model.stan"), "{args}");
    let mut build_files: Vec<String> = fs::read_dir(project.dir.join("build"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    build_files.sort();
    assert_eq!(build_files, vec!["gps.stanfunctions", "model.stan"]);
}

#[test]
fn build_output_may_name_a_directory() {
    let project = setup();
    project.write_project_file("model.laplace", "data {\n  int n;\n}\n");
    fs::create_dir_all(project.dir.join("out")).unwrap();

    let out = project.run(&["build", "model.laplace", "--output", "out"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(project.project_file_exists("out/model.stan"));

    let out = project.run(&["build", "model.laplace", "--output", "fresh/"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(project.project_file_exists("fresh/model.stan"));
}

#[test]
fn add_rejects_a_hyphenated_package_name() {
    let project = setup();
    let out = project.run(&["add", "laplace-splines"]);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("invalid package name `laplace-splines`"),
        "{}",
        stderr(&out)
    );
    assert!(!project.project_file_exists("laplace.toml"));
}

#[test]
fn build_rejects_an_import_pinned_to_a_version_the_lock_does_not_hold() {
    let project = setup();
    project.write_package("gps", "1.0.0", &["rbf_cov"], GPS_STAN);
    assert!(project.run(&["add", "gps"]).status.success());
    project.write_project_file(
        "model.laplace",
        "library {\n  import gps@9.9.9\n}\nmodel {\n}\n",
    );

    let out = project.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains("imports `gps@9.9.9`, but laplace.lock pins `gps` at 1.0.0"),
        "{err}"
    );
}

#[test]
fn doc_prints_every_overload() {
    let project = setup();
    project.write_package(
        "kinetics",
        "1.0.0",
        &["hill"],
        "// @laplace\n// @brief Hill curve.\nreal hill(real x) {\n  return x;\n}\nvector hill(vector x) {\n  return x;\n}\n",
    );
    assert!(project.run(&["add", "kinetics"]).status.success());

    let out = project.run(&["doc", "kinetics::hill"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("kinetics::hill has 2 overloads"), "{text}");
    assert!(text.contains("kinetics::hill(x: real) -> real"), "{text}");
    assert!(
        text.contains("kinetics::hill(x: vector) -> vector"),
        "{text}"
    );
    assert!(text.contains("Hill curve."), "{text}");
}

// ---------------------------------------------------------------------------
// Patch 1 session 1: `pub` visibility, naming hygiene, provenance
// ---------------------------------------------------------------------------

/// A `.laplacelib` package whose public `mean_` calls a private
/// `sum_values`, written verbatim (not through `mark_pub`) so the test
/// shows the dialect a library author actually writes.
const STATS_LIB: &str = r#"// @laplace
// @brief Arithmetic mean of a vector.
// @param x The vector to average.
// @return The mean of `x`.
pub real mean_(vector x) {
  return sum_values(x) / num_elements(x);
}

real sum_values(vector x) {
  return sum(x);
}
"#;

/// Type-check a generated file with a real `stanc`, if there is one.
/// Prints a note and passes when there is not: a missing Stan toolchain
/// is not a failure of laplace.
fn stanc_accepts(p: &Project, file: &str) {
    let Some(stanc) = real_stanc() else {
        eprintln!("note: no `stanc` on PATH -- skipping the stanc acceptance check for {file}");
        return;
    };
    let out = Command::new(&stanc)
        .arg("--include-paths=build")
        .arg(p.dir.join(file))
        .current_dir(&p.dir)
        .output()
        .expect("failed to run stanc");
    assert!(
        out.status.success(),
        "stanc rejected {file}:\n{}\n--- generated ---\n{}",
        String::from_utf8_lossy(&out.stderr),
        p.read_project_file(file),
    );
}

fn stats_lib_project() -> Project {
    let p = setup();
    p.write_package_file(
        "stats",
        "1.0.0",
        "laplace.toml",
        "name = \"stats\"\nversion = \"1.0.0\"\n",
    );
    p.write_package_file("stats", "1.0.0", "stats.laplacelib", STATS_LIB);
    p
}

#[test]
fn a_pub_item_is_callable_and_a_private_one_is_not() {
    let p = stats_lib_project();
    p.write_project_file(
        "model.laplace",
        "library {\n  import stats\n}\n\ndata {\n  vector[3] y;\n}\nmodel {\n  real m = stats::mean_(y);\n}\n",
    );
    assert!(p.run(&["add", "stats"]).status.success());
    assert!(p.run(&["install"]).status.success());

    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");
    assert!(stan.contains("real stats__mean_(vector x)"), "{stan}");
    assert!(stan.contains("stats__mean_(y)"), "{stan}");
    stanc_accepts(&p, "build/model.stan");

    // The private helper is reachable only from inside the package.
    p.write_project_file(
        "model.laplace",
        "library {\n  import stats\n}\n\ndata {\n  vector[3] y;\n}\nmodel {\n  real s = stats::sum_values(y);\n}\n",
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains("`stats::sum_values` is private to package `stats`"),
        "{err}"
    );
    assert!(err.contains("model.laplace:9:12"), "{err}");
    assert!(
        err.contains("only items marked `pub` can be used outside their package"),
        "{err}"
    );
}

#[test]
fn a_private_item_called_from_a_pub_item_in_the_same_package_builds_and_both_are_emitted() {
    let p = stats_lib_project();
    p.write_project_file(
        "model.laplace",
        "library {\n  import stats\n}\n\ndata {\n  vector[3] y;\n}\nmodel {\n  real m = stats::mean_(y);\n}\n",
    );
    assert!(p.run(&["add", "stats"]).status.success());
    assert!(p.run(&["install"]).status.success());
    assert!(p.run(&["build", "model.laplace"]).status.success());

    let stan = p.read_project_file("build/model.stan");
    // Both definitions are emitted: private is an access rule, not hiding.
    assert!(stan.contains("real stats__mean_(vector x)"), "{stan}");
    assert!(stan.contains("real stats__sum_values(vector x)"), "{stan}");
    // And the internal unqualified call was mangled to its own package.
    assert!(
        stan.contains("return stats__sum_values(x) / num_elements(x);"),
        "{stan}"
    );
    // `pub` never reaches the output.
    assert!(!stan.contains("pub "), "{stan}");
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn a_library_calling_another_librarys_private_item_is_an_error() {
    let p = stats_lib_project();
    // `regression` imports `stats` and reaches for its private helper.
    p.write_package_file(
        "regression",
        "1.0.0",
        "laplace.toml",
        "name = \"regression\"\nversion = \"1.0.0\"\n\n[dependencies]\nstats = \"^1.0\"\n",
    );
    p.write_package_file(
        "regression",
        "1.0.0",
        "regression.laplacelib",
        "library {\n  import stats\n}\n\npub real total(vector x) {\n  return stats::sum_values(x);\n}\n",
    );
    p.write_project_file(
        "model.laplace",
        "library {\n  import regression\n}\n\ndata {\n  vector[3] y;\n}\nmodel {\n  real t = regression::total(y);\n}\n",
    );

    assert!(p.run(&["add", "regression"]).status.success());
    assert!(p.run(&["install"]).status.success());
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains("`stats::sum_values` is private to package `stats`"),
        "{err}"
    );
    // The location is inside the library, named the way its author would
    // open it -- not a line in the user's own model.
    assert!(
        err.contains("regression/regression.laplacelib:6:10"),
        "{err}"
    );
}

#[test]
fn an_old_style_plain_stan_library_keeps_working_with_exports() {
    let p = setup();
    p.write_package("stats", "1.0.0", &["mean_"], STATS_STAN);
    p.write_project_file(
        "model.laplace",
        "library {\n  import stats\n}\n\ndata {\n  vector[3] y;\n}\nmodel {\n  real m = stats::mean_(y);\n}\n",
    );
    assert!(p.run(&["add", "stats"]).status.success());
    assert!(p.run(&["install"]).status.success());

    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");
    assert!(stan.contains("real stats__mean_(vector x)"), "{stan}");
    assert!(
        stan.contains("// stats v1.0.0 (pub) -- stats/stats.stan:5"),
        "{stan}"
    );
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn listing_a_laplacelib_item_in_exports_says_to_use_pub_instead() {
    let p = setup();
    p.write_package_file(
        "stats",
        "1.0.0",
        "laplace.toml",
        "name = \"stats\"\nversion = \"1.0.0\"\nexports = [\"mean_\"]\n",
    );
    p.write_package_file("stats", "1.0.0", "stats.laplacelib", STATS_LIB);
    p.write_project_file(
        "model.laplace",
        "library {\n  import stats\n}\nmodel {\n}\n",
    );
    assert!(p.run(&["add", "stats"]).status.success());
    assert!(p.run(&["install"]).status.success());

    // The manifest and the source disagree about what is public, which
    // is caught when the package is loaded for a build.
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("lists `mean_` under `exports`"), "{err}");
    assert!(err.contains("`.laplacelib`"), "{err}");
    assert!(err.contains("write `pub` in front of `mean_`"), "{err}");
}

#[test]
fn marking_only_one_overload_pub_is_an_error() {
    let p = setup();
    p.write_package_file(
        "over",
        "1.0.0",
        "laplace.toml",
        "name = \"over\"\nversion = \"1.0.0\"\n",
    );
    p.write_package_file(
        "over",
        "1.0.0",
        "over.laplacelib",
        "pub real h(real x) {\n  return x;\n}\n\nreal h(vector x) {\n  return x[1];\n}\n",
    );
    p.write_project_file("model.laplace", "library {\n  import over\n}\nmodel {\n}\n");
    assert!(p.run(&["add", "over"]).status.success());
    assert!(p.run(&["install"]).status.success());

    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("`h`"), "{err}");
    assert!(err.contains("mark every definition"), "{err}");
}

#[test]
fn a_double_underscore_identifier_in_a_laplace_file_is_rejected() {
    let p = setup();
    p.write_project_file(
        "model.laplace",
        "data {\n  int N;\n}\nparameters {\n  real my__theta;\n}\nmodel {\n}\n",
    );

    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("`my__theta` contains `__`"), "{err}");
    assert!(err.contains("model.laplace:5:8"), "{err}");
    assert!(err.contains("my_theta"), "{err}");
}

#[test]
fn a_double_underscore_identifier_in_a_laplacelib_is_rejected() {
    let p = setup();
    p.write_package_file(
        "bad",
        "1.0.0",
        "laplace.toml",
        "name = \"bad\"\nversion = \"1.0.0\"\n",
    );
    p.write_package_file(
        "bad",
        "1.0.0",
        "bad.laplacelib",
        "pub real f(real x) {\n  return my__helper(x);\n}\n",
    );

    let out = p.run(&["add", "bad"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("`my__helper` contains `__`"), "{err}");
    assert!(err.contains("bad.laplacelib:2:10"), "{err}");
}

#[test]
fn doc_hides_a_private_laplacelib_item_but_shows_a_pub_one() {
    let p = stats_lib_project();
    assert!(p.run(&["add", "stats"]).status.success());

    let out = p.run(&["doc", "stats::mean_"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stdout(&out).contains("Arithmetic mean of a vector."),
        "{}",
        stdout(&out)
    );

    let out = p.run(&["doc", "stats::sum_values"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains("`stats::sum_values` is private to package `stats`"),
        "{err}"
    );
    assert!(err.contains("only items marked `pub`"), "{err}");
}

#[test]
fn provenance_comments_name_the_package_version_visibility_file_and_line() {
    let p = stats_lib_project();
    p.write_project_file(
        "model.laplace",
        "library {\n  import stats\n}\n\ndata {\n  vector[3] y;\n}\nmodel {\n  real m = stats::mean_(y);\n}\n",
    );
    assert!(p.run(&["add", "stats"]).status.success());
    assert!(p.run(&["install"]).status.success());
    assert!(p.run(&["build", "model.laplace"]).status.success());

    let stan = p.read_project_file("build/model.stan");
    // `mean_`'s definition is on line 5 of stats.laplacelib, `sum_values`
    // on line 9 -- the lines in the *source*, not in the stripped body.
    assert!(
        stan.contains("// stats v1.0.0 (pub) -- stats/stats.laplacelib:5"),
        "{stan}"
    );
    assert!(
        stan.contains("// stats v1.0.0 (private) -- stats/stats.laplacelib:9"),
        "{stan}"
    );
    // No absolute path leaked into the output, or builds would differ
    // between machines.
    assert!(!stan.contains(".laplace/packages"), "{stan}");
}

#[test]
fn a_build_is_byte_identical_when_repeated_and_check_agrees() {
    let p = stats_lib_project();
    p.write_project_file(
        "model.laplace",
        "library {\n  import stats\n}\n\ndata {\n  vector[3] y;\n}\nmodel {\n  real m = stats::mean_(y);\n}\n",
    );
    assert!(p.run(&["add", "stats"]).status.success());
    assert!(p.run(&["install"]).status.success());

    assert!(p.run(&["build", "model.laplace"]).status.success());
    let first = p.read_project_file("build/model.stan");
    assert!(p.run(&["build", "model.laplace"]).status.success());
    assert_eq!(first, p.read_project_file("build/model.stan"));
    assert!(p
        .run(&["build", "model.laplace", "--check"])
        .status
        .success());
}

#[test]
fn a_project_that_imports_nothing_is_passed_through_byte_for_byte() {
    let p = setup();
    let source = "data {\n  int<lower=1> N;\n  vector[N] y;\n}\nparameters {\n  real mu;\n}\nmodel {\n  y ~ normal(mu, 1);\n}\n";
    p.write_project_file("model.laplace", source);

    assert!(p.run(&["build", "model.laplace"]).status.success());
    assert_eq!(p.read_project_file("build/model.stan"), source);
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn each_package_in_a_chain_resolves_its_own_private_helper_of_the_same_name() {
    let p = setup();
    // Both `stats` and `regression` define a private `helper` with an
    // identical signature. Each must call its own.
    p.write_package_file(
        "stats",
        "1.0.0",
        "laplace.toml",
        "name = \"stats\"\nversion = \"1.0.0\"\n",
    );
    p.write_package_file(
        "stats",
        "1.0.0",
        "stats.laplacelib",
        "real helper(real x) {\n  return x + 1;\n}\n\npub real mean_(real x) {\n  return helper(x);\n}\n",
    );
    p.write_package_file(
        "regression",
        "1.0.0",
        "laplace.toml",
        "name = \"regression\"\nversion = \"1.0.0\"\n\n[dependencies]\nstats = \"^1.0\"\n",
    );
    p.write_package_file(
        "regression",
        "1.0.0",
        "regression.laplacelib",
        "library {\n  import stats\n}\n\nreal helper(real x) {\n  return x * 2;\n}\n\npub real fit(real x) {\n  return helper(stats::mean_(x));\n}\n",
    );
    p.write_project_file(
        "model.laplace",
        "library {\n  import regression\n}\n\nmodel {\n  real f = regression::fit(1.0);\n}\n",
    );

    assert!(p.run(&["add", "regression"]).status.success());
    assert!(p.run(&["install"]).status.success());
    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));

    let stan = p.read_project_file("build/model.stan");
    assert!(stan.contains("real stats__helper(real x)"), "{stan}");
    assert!(stan.contains("real regression__helper(real x)"), "{stan}");
    assert!(stan.contains("return stats__helper(x);"), "{stan}");
    assert!(
        stan.contains("return regression__helper(stats__mean_(x));"),
        "{stan}"
    );
    stanc_accepts(&p, "build/model.stan");
}

// ---------------------------------------------------------------------------
// Patch 1 session 2: functions as arguments (monomorphization)
// ---------------------------------------------------------------------------

/// A library with two higher-order functions: a scalar map, and one that
/// needs `@wait` because it cannot know the bound function's return size.
const TRANSFORMS_LIB: &str = r#"// @laplace
// @brief Apply a scalar map to every element of a vector.
pub vector map_each(vector x, func(real) -> real f) {
  vector[num_elements(x)] out;
  for (i in 1:num_elements(x)) {
    out[i] = f(x[i]);
  }
  return out;
}

// @laplace
// @brief Stack the results of a vector-valued map as matrix rows.
pub matrix expand_rows(vector x, func(real) -> vector f) {
  matrix[num_elements(x), @wait(f).size] out;
  for (i in 1:num_elements(x)) {
    @wait(f) row = f(x[i]);
    out[i] = row';
  }
  return out;
}

real scale_(real x) {
  return 2 * x;
}

pub vector doubled(vector x) {
  return map_each(x, scale_);
}
"#;

fn transforms_project() -> Project {
    let p = setup();
    p.write_package_file(
        "transforms",
        "1.0.0",
        "laplace.toml",
        "name = \"transforms\"\nversion = \"1.0.0\"\n",
    );
    p.write_package_file(
        "transforms",
        "1.0.0",
        "transforms.laplacelib",
        TRANSFORMS_LIB,
    );
    p
}

#[test]
fn a_user_higher_order_function_is_specialized_and_nothing_generic_survives() {
    let p = setup();
    p.write_project_file(
        "model.laplace",
        concat!(
            "functions {\n",
            "  real add_one(real x) {\n    return x + 1;\n  }\n",
            "  real apply_twice(real x, func(real) -> real f) {\n",
            "    real a = f(x);\n",
            "    return f(a);\n",
            "  }\n",
            "}\n",
            "transformed data {\n",
            "  real r = apply_twice(5, add_one);\n",
            "}\n",
            "model {\n}\n",
        ),
    );

    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");

    assert!(
        stan.contains(
            "real apply_twice__add_one(real x) {\n  real a = add_one(x);\n  return add_one(a);\n}"
        ),
        "{stan}"
    );
    assert!(stan.contains("real r = apply_twice__add_one(5);"), "{stan}");
    assert!(
        stan.contains("// monomorphized: apply_twice with f = add_one"),
        "{stan}"
    );
    // No laplace-only syntax reaches the output.
    assert!(!stan.contains("func("), "{stan}");
    assert!(!stan.contains("@wait"), "{stan}");
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn a_library_hof_binds_a_user_function_a_library_function_and_a_private_one() {
    let p = transforms_project();
    p.write_project_file(
        "model.laplace",
        concat!(
            "library {\n  import transforms\n}\n",
            "functions {\n",
            "  real softplus(real x) {\n    return log1p_exp(x);\n  }\n",
            "  vector[2] to_pair(real x) {\n    return [x, x * 2]';\n  }\n",
            "}\n",
            "data {\n  int<lower=1> N;\n  vector[N] y;\n}\n",
            "transformed data {\n",
            "  vector[N] s = transforms::map_each(y, softplus);\n",
            "  matrix[N, 2] pairs = transforms::expand_rows(y, to_pair);\n",
            "  vector[N] d = transforms::doubled(y);\n",
            "}\n",
            "model {\n}\n",
        ),
    );
    assert!(p.run(&["add", "transforms"]).status.success());
    assert!(p.run(&["install"]).status.success());
    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");

    // A user function bound into a library HOF.
    assert!(
        stan.contains("vector transforms__map_each__softplus(vector x)"),
        "{stan}"
    );
    assert!(stan.contains("out[i] = softplus(x[i]);"), "{stan}");
    // The library's own private function bound into its own HOF.
    assert!(
        stan.contains("vector transforms__map_each__transforms__scale_(vector x)"),
        "{stan}"
    );
    assert!(
        stan.contains("out[i] = transforms__scale_(x[i]);"),
        "{stan}"
    );
    // That one is called from inside a function body, so it is declared first.
    let declaration = stan
        .find("vector transforms__map_each__transforms__scale_(vector x);")
        .expect("forward declaration");
    let caller = stan.find("vector transforms__doubled(vector x)").unwrap();
    assert!(declaration < caller, "{stan}");

    // `@wait` resolved against a literal-sized return type.
    assert!(stan.contains("matrix[num_elements(x), 2] out;"), "{stan}");
    assert!(stan.contains("vector[2] row = to_pair(x[i]);"), "{stan}");
    // ...and the size annotation stripped from the function itself.
    assert!(stan.contains("vector to_pair(real x)"), "{stan}");
    assert!(!stan.contains("vector[2] to_pair"), "{stan}");

    assert!(!stan.contains("func("), "{stan}");
    assert!(!stan.contains("@wait"), "{stan}");
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn the_same_binding_twice_emits_one_copy_and_two_bindings_emit_two() {
    let p = setup();
    p.write_project_file(
        "model.laplace",
        concat!(
            "functions {\n",
            "  real inc(real x) {\n    return x + 1;\n  }\n",
            "  real dbl(real x) {\n    return 2 * x;\n  }\n",
            "  real twice(real x, func(real) -> real f) {\n    return f(f(x));\n  }\n",
            "}\n",
            "transformed data {\n",
            "  real a = twice(1, inc);\n",
            "  real b = twice(2, inc);\n",
            "  real c = twice(3, dbl);\n",
            "}\n",
            "model {\n}\n",
        ),
    );
    assert!(p.run(&["build", "model.laplace"]).status.success());
    let stan = p.read_project_file("build/model.stan");

    assert_eq!(
        stan.matches("real twice__inc(real x) {").count(),
        1,
        "{stan}"
    );
    assert_eq!(
        stan.matches("real twice__dbl(real x) {").count(),
        1,
        "{stan}"
    );
    assert!(stan.contains("real a = twice__inc(1);"), "{stan}");
    assert!(stan.contains("real b = twice__inc(2);"), "{stan}");
    assert!(stan.contains("real c = twice__dbl(3);"), "{stan}");
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn a_parameter_dependent_return_size_works_in_the_direct_call_pattern() {
    let p = setup();
    p.write_project_file(
        "model.laplace",
        concat!(
            "functions {\n",
            "  vector[K] basis(real t, int K) {\n    return rep_vector(t, K);\n  }\n",
            "  vector first_basis(real t, int k, func(real, int) -> vector f) {\n",
            "    @wait(f) r = f(t, k);\n",
            "    return r;\n",
            "  }\n",
            "}\n",
            "transformed data {\n",
            "  vector[4] b = first_basis(1.0, 4, basis);\n",
            "}\n",
            "model {\n}\n",
        ),
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");
    assert!(stan.contains("vector[(k)] r = basis(t, k);"), "{stan}");
    assert!(stan.contains("vector basis(real t, int K)"), "{stan}");
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn a_bound_function_without_a_return_size_errors_at_the_call_site() {
    let p = transforms_project();
    p.write_project_file(
        "model.laplace",
        concat!(
            "library {\n  import transforms\n}\n",
            "functions {\n",
            "  vector to_pair(real x) {\n    return [x, x * 2]';\n  }\n",
            "}\n",
            "data {\n  vector[3] y;\n}\n",
            "transformed data {\n",
            "  matrix[3, 2] pairs = transforms::expand_rows(y, to_pair);\n",
            "}\n",
            "model {\n}\n",
        ),
    );
    assert!(p.run(&["add", "transforms"]).status.success());
    assert!(p.run(&["install"]).status.success());

    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("needs the return size of `to_pair`"), "{err}");
    assert!(err.contains("@wait(f)"), "{err}");
    assert!(err.contains("annotate the return type"), "{err}");
    // Reported where the binding was made, in the user's own file.
    assert!(err.contains("model.laplace:13"), "{err}");
}

#[test]
fn binding_a_private_library_function_from_the_project_is_refused() {
    let p = transforms_project();
    p.write_project_file(
        "model.laplace",
        concat!(
            "library {\n  import transforms\n}\n",
            "data {\n  vector[3] y;\n}\n",
            "transformed data {\n",
            "  vector[3] d = transforms::map_each(y, transforms::scale_);\n",
            "}\n",
            "model {\n}\n",
        ),
    );
    assert!(p.run(&["add", "transforms"]).status.success());
    assert!(p.run(&["install"]).status.success());

    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("`transforms::scale_` is private"), "{err}");
    assert!(err.contains("only items marked `pub`"), "{err}");
}

#[test]
fn every_v1_restriction_on_functional_parameters_has_its_own_error() {
    let cases: &[(&str, &str)] = &[
        (
            "stored rather than called",
            "  real h(func(real) -> real f) {\n    real g = f;\n    return g;\n  }\n",
        ),
        (
            "called with the wrong arity",
            "  real h(real x, func(real) -> real f) {\n    return f(x, x);\n  }\n",
        ),
        (
            "a func inside a func",
            "  real h(func(func(real) -> real) -> real f) {\n    return 1;\n  }\n",
        ),
        (
            "an array return type",
            "  real h(func(real) -> array[] real f) {\n    return 1;\n  }\n",
        ),
        (
            "a recursive higher-order function",
            "  real inc(real x) {\n    return x + 1;\n  }\n  real h(real x, func(real) -> real f) {\n    if (x > 0) return h(x - 1, inc);\n    return f(x);\n  }\n",
        ),
    ];

    for (what, functions) in cases {
        let p = setup();
        p.write_project_file(
            "model.laplace",
            &format!("functions {{\n{functions}}}\nmodel {{\n}}\n"),
        );
        let out = p.run(&["build", "model.laplace"]);
        assert!(!out.status.success(), "{what} should not build");
        let err = stderr(&out);
        assert!(err.starts_with("error: "), "{what}: {err}");
        assert!(err.contains("help: "), "{what} needs a help line: {err}");
    }
}

#[test]
fn binding_a_stan_builtin_suggests_writing_a_wrapper() {
    let p = setup();
    p.write_project_file(
        "model.laplace",
        concat!(
            "functions {\n",
            "  real twice(real x, func(real) -> real f) {\n    return f(f(x));\n  }\n",
            "}\n",
            "transformed data {\n  real r = twice(1, exp);\n}\n",
            "model {\n}\n",
        ),
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("`exp` is a Stan built-in"), "{err}");
    assert!(
        err.contains("real exp_(real x) { return exp(x); }"),
        "{err}"
    );
}

#[test]
fn a_higher_order_function_that_is_never_called_is_not_emitted() {
    let p = setup();
    p.write_project_file(
        "model.laplace",
        concat!(
            "functions {\n",
            "  real used(real x) {\n    return x;\n  }\n",
            "  real twice(real x, func(real) -> real f) {\n    return f(f(x));\n  }\n",
            "}\n",
            "transformed data {\n  real r = used(1);\n}\n",
            "model {\n}\n",
        ),
    );
    assert!(p.run(&["build", "model.laplace"]).status.success());
    let stan = p.read_project_file("build/model.stan");
    assert!(!stan.contains("twice"), "{stan}");
    assert!(stan.contains("real used(real x)"), "{stan}");
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn a_project_using_no_functional_parameters_is_unchanged_by_this_feature() {
    let p = stats_lib_project();
    p.write_project_file(
        "model.laplace",
        "library {\n  import stats\n}\n\ndata {\n  vector[3] y;\n}\nmodel {\n  real m = stats::mean_(y);\n}\n",
    );
    assert!(p.run(&["add", "stats"]).status.success());
    assert!(p.run(&["install"]).status.success());
    assert!(p.run(&["build", "model.laplace"]).status.success());

    let stan = p.read_project_file("build/model.stan");
    assert!(!stan.contains("monomorphized"), "{stan}");
    assert!(!stan.contains("laplace: specialized"), "{stan}");
    // And a second build agrees byte for byte.
    assert!(p
        .run(&["build", "model.laplace", "--check"])
        .status
        .success());
}

#[test]
fn specialized_functions_stay_in_the_stan_file_in_split_mode() {
    let p = transforms_project();
    p.write_project_file(
        "model.laplace",
        concat!(
            "library {\n  import transforms\n}\n",
            "functions {\n",
            "  real softplus(real x) {\n    return log1p_exp(x);\n  }\n",
            "}\n",
            "data {\n  vector[3] y;\n}\n",
            "transformed data {\n",
            "  vector[3] s = transforms::map_each(y, softplus);\n",
            "}\n",
            "model {\n}\n",
        ),
    );
    assert!(p.run(&["add", "transforms"]).status.success());
    assert!(p.run(&["install"]).status.success());
    let out = p.run(&["build", "model.laplace", "--split-functions"]);
    assert!(out.status.success(), "{}", stderr(&out));

    let stan = p.read_project_file("build/model.stan");
    let functions = p.read_project_file("build/transforms.stanfunctions");
    // The copy calls a function defined in the model, so it has to live
    // there, not in the package's own file.
    assert!(
        stan.contains("vector transforms__map_each__softplus(vector x)"),
        "{stan}"
    );
    assert!(!functions.contains("map_each__softplus"), "{functions}");
    assert!(!functions.contains("func("), "{functions}");
    stanc_accepts(&p, "build/model.stan");
}

// ---------------------------------------------------------------------------
// Patch 1 session 3: block-spanning templates (@template / @use)
// ---------------------------------------------------------------------------

const TEMPLATES_LIB: &str = r#"// @laplace
// @brief Non-centred parameterization for a vector of coefficients.
pub @template ncp($name: ident, $N: expr) {
  parameters {
    vector[$N] ${name}_raw;
    real<lower=0> ${name}_sigma;
  }
  transformed parameters {
    vector[$N] $name = ${name}_sigma * ${name}_raw;
  }
  model {
    ${name}_raw ~ std_normal();
    ${name}_sigma ~ exponential(1);
  }
}

pub @template observation($y: ident, $mu: expr, $sigma: expr) {
  model {
    $y ~ lognormal($mu, $sigma);
  }
  generated quantities {
    real ${y}_rep = lognormal_rng($mu, $sigma);
  }
}

pub @template scaled($out: ident, $src: expr) {
  transformed parameters {
    real $out = half($src);
  }
}

@template internal_only($n: ident) {
  parameters {
    real $n;
  }
}

real half(real x) {
  return x / 2;
}
"#;

fn templates_project() -> Project {
    let p = setup();
    p.write_package_file(
        "stats",
        "1.0.0",
        "laplace.toml",
        "name = \"stats\"\nversion = \"1.0.0\"\n",
    );
    p.write_package_file("stats", "1.0.0", "stats.laplacelib", TEMPLATES_LIB);
    p
}

/// A project with `stats` installed and `model.laplace` written.
fn with_model(p: &Project, model: &str) {
    p.write_project_file("model.laplace", model);
    assert!(p.run(&["add", "stats"]).status.success());
    assert!(p.run(&["install"]).status.success());
}

#[test]
fn a_template_expands_into_its_blocks_and_creates_the_ones_that_are_missing() {
    let p = templates_project();
    with_model(
        &p,
        concat!(
            "library {\n  import stats\n}\n",
            "\n",
            "@use stats::ncp(theta, K);\n",
            "\n",
            "data {\n  int<lower=1> K;\n  vector[K] y;\n}\n",
            "model {\n  y ~ normal(theta[1], 1);\n}\n",
        ),
    );

    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");

    // The pieces land in their blocks, before the model's own content.
    assert!(
        stan.contains("parameters {\n  // begin @use stats::ncp(theta, K) -- model.laplace:5\n  vector[K] theta_raw;\n  real<lower=0> theta_sigma;\n  // end @use stats::ncp\n"),
        "{stan}"
    );
    // `parameters` and `transformed parameters` did not exist and were
    // created, in Stan's own block order.
    let data = stan.find("data {").unwrap();
    let params = stan.find("parameters {").unwrap();
    let tparams = stan.find("transformed parameters {").unwrap();
    let model = stan.find("model {").unwrap();
    assert!(
        data < params && params < tparams && tparams < model,
        "{stan}"
    );
    assert!(
        stan.contains("vector[K] theta = theta_sigma * theta_raw;"),
        "{stan}"
    );

    // No laplace syntax survives. `@use` still appears inside the
    // provenance comments, which is exactly what they are for.
    assert!(
        !stan
            .lines()
            .any(|line| line.trim_start().starts_with("@use")),
        "{stan}"
    );
    assert!(!stan.contains("@template"), "{stan}");
    assert!(!stan.contains('$'), "{stan}");
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn two_uses_write_into_the_same_blocks_in_use_order_before_the_users_content() {
    let p = templates_project();
    with_model(
        &p,
        concat!(
            "library {\n  import stats\n}\n",
            "@use stats::ncp(theta, K);\n",
            "@use stats::observation(y, mu + theta[1], sigma);\n",
            "data {\n  int<lower=1> K;\n  vector[K] y;\n}\n",
            "parameters {\n  real mu;\n  real<lower=0> sigma;\n}\n",
            "model {\n  mu ~ normal(0, 1);\n}\n",
        ),
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");

    let ncp = stan.find("theta_raw ~ std_normal();").unwrap();
    let observation = stan.find("y ~ lognormal(").unwrap();
    let user = stan.find("mu ~ normal(0, 1);").unwrap();
    assert!(ncp < observation, "`@use` order decides: {stan}");
    assert!(
        observation < user,
        "pieces come before the model's own: {stan}"
    );

    // The model's own `parameters` content stays after the expansion.
    let theta_raw = stan.find("vector[K] theta_raw;").unwrap();
    let mu = stan.find("real mu;").unwrap();
    assert!(theta_raw < mu, "{stan}");
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn an_expr_argument_is_parenthesized_and_a_plain_one_is_not() {
    let p = templates_project();
    with_model(
        &p,
        concat!(
            "library {\n  import stats\n}\n",
            "@use stats::observation(y, mu + theta, sigma);\n",
            "data {\n  vector[3] y;\n}\n",
            "parameters {\n  real mu;\n  real theta;\n  real<lower=0> sigma;\n}\n",
        ),
    );
    assert!(p.run(&["build", "model.laplace"]).status.success());
    let stan = p.read_project_file("build/model.stan");
    assert!(
        stan.contains("y ~ lognormal((mu + theta), sigma);"),
        "{stan}"
    );
    assert!(
        stan.contains("real y_rep = lognormal_rng((mu + theta), sigma);"),
        "{stan}"
    );
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn one_template_used_twice_with_different_names_expands_twice() {
    let p = templates_project();
    with_model(
        &p,
        concat!(
            "library {\n  import stats\n}\n",
            "@use stats::ncp(theta, K);\n",
            "@use stats::ncp(beta, P);\n",
            "data {\n  int<lower=1> K;\n  int<lower=1> P;\n}\n",
        ),
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");
    for name in ["theta", "beta"] {
        assert!(
            stan.contains(&format!("{name}_raw ~ std_normal();")),
            "{stan}"
        );
        assert!(
            stan.contains(&format!("real<lower=0> {name}_sigma;")),
            "{stan}"
        );
    }
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn a_template_body_calls_its_own_packages_private_helper() {
    let p = templates_project();
    with_model(
        &p,
        concat!(
            "library {\n  import stats\n}\n",
            "@use stats::scaled(mu_half, mu);\n",
            "parameters {\n  real mu;\n}\n",
        ),
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");
    // `half` is private to `stats`, and the call resolves in the
    // defining package's scope even though the code now lives here.
    assert!(stan.contains("real mu_half = stats__half(mu);"), "{stan}");
    assert!(stan.contains("real stats__half(real x)"), "{stan}");
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn two_uses_declaring_the_same_name_collide_and_so_does_one_with_the_model() {
    let p = templates_project();

    with_model(
        &p,
        "library {\n  import stats\n}\n@use stats::ncp(theta, K);\n@use stats::ncp(theta, P);\ndata {\n  int K;\n}\n",
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("`theta_raw` is declared twice"), "{err}");
    assert!(err.contains("model.laplace:4"), "{err}");
    assert!(err.contains("model.laplace:5"), "{err}");

    // ...and against a declaration the user wrote by hand.
    p.write_project_file(
        "model.laplace",
        "library {\n  import stats\n}\n@use stats::ncp(theta, K);\nparameters {\n  vector[K] theta_raw;\n}\n",
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("`theta_raw` is declared twice"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn a_private_template_cannot_be_used_from_a_model() {
    let p = templates_project();
    with_model(
        &p,
        "library {\n  import stats\n}\n@use stats::internal_only(z);\n",
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("`stats::internal_only` is private"), "{err}");
    assert!(err.contains("only templates marked `pub`"), "{err}");
}

#[test]
fn every_definition_time_template_error_has_its_own_message() {
    let cases: &[(&str, &str, &str)] = &[
        (
            "a functions block",
            "pub @template t($n: ident) {\n  functions {\n    real f() { return 1; }\n  }\n}\n",
            "not a Stan program block",
        ),
        (
            "a duplicated block",
            "pub @template t($n: ident) {\n  model {\n    $n ~ std_normal();\n  }\n  model {\n    $n ~ std_normal();\n  }\n}\n",
            "two `model` pieces",
        ),
        (
            "an undeclared placeholder",
            "pub @template t($n: ident) {\n  model {\n    $n ~ normal($mu, 1);\n  }\n}\n",
            "which its header does not declare",
        ),
        (
            "a fixed-name declaration",
            "pub @template t($n: ident) {\n  transformed parameters {\n    real tmp = 1;\n    real $n = tmp;\n  }\n}\n",
            "with a fixed name",
        ),
        (
            "reaching for a model variable",
            "pub @template t($n: ident) {\n  model {\n    $n ~ normal(mu, 1);\n  }\n}\n",
            "which it does not declare",
        ),
        (
            "a nested @use",
            "pub @template t($n: ident) {\n  model {\n    @use other::x($n);\n  }\n}\n",
            "nested `@use`",
        ),
        (
            "an expr placeholder naming a variable",
            "pub @template t($n: expr) {\n  parameters {\n    real ${n}_raw;\n  }\n}\n",
            "not an `ident`",
        ),
        (
            "a statement outside a block piece",
            "pub @template t($n: ident) {\n  $n ~ std_normal();\n}\n",
            "holds only Stan block pieces",
        ),
    ];

    for (what, library, expected) in cases {
        let p = setup();
        p.write_package_file(
            "bad",
            "1.0.0",
            "laplace.toml",
            "name = \"bad\"\nversion = \"1.0.0\"\n",
        );
        p.write_package_file("bad", "1.0.0", "bad.laplacelib", library);

        let out = p.run(&["add", "bad"]);
        assert!(!out.status.success(), "{what} should not be accepted");
        let err = stderr(&out);
        assert!(
            err.contains(expected),
            "{what}: expected {expected:?} in\n{err}"
        );
        assert!(
            err.contains("bad.laplacelib:"),
            "{what} needs a location: {err}"
        );
        assert!(err.contains("help: "), "{what} needs a help line: {err}");
    }
}

#[test]
fn an_unused_placeholder_warns_without_failing_the_build() {
    let p = setup();
    p.write_package_file(
        "stats",
        "1.0.0",
        "laplace.toml",
        "name = \"stats\"\nversion = \"1.0.0\"\n",
    );
    p.write_package_file(
        "stats",
        "1.0.0",
        "stats.laplacelib",
        "pub @template t($used: ident, $spare: expr) {\n  parameters {\n    real $used;\n  }\n}\n",
    );
    with_model(
        &p,
        "library {\n  import stats\n}\n@use stats::t(theta, 1);\nmodel {\n}\n",
    );

    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("warning:"), "{err}");
    assert!(err.contains("$spare"), "{err}");
    assert!(p
        .read_project_file("build/model.stan")
        .contains("real theta;"));
}

#[test]
fn a_use_argument_may_not_call_a_higher_order_function() {
    let p = templates_project();
    with_model(
        &p,
        concat!(
            "library {\n  import stats\n}\n",
            "functions {\n",
            "  real inc(real x) {\n    return x + 1;\n  }\n",
            "  real twice(real x, func(real) -> real f) {\n    return f(f(x));\n  }\n",
            "}\n",
            "@use stats::scaled(z, twice(1, inc));\n",
        ),
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("called inside a `@use` argument"), "{err}");
    assert!(err.contains("assign"), "{err}");
}

#[test]
fn a_project_with_no_use_statements_is_unaffected_by_templates() {
    // `stats` defines templates this model never uses.
    let p = setup();
    p.write_package_file(
        "stats",
        "1.0.0",
        "laplace.toml",
        "name = \"stats\"\nversion = \"1.0.0\"\n",
    );
    p.write_package_file(
        "stats",
        "1.0.0",
        "stats.laplacelib",
        &format!("{TEMPLATES_LIB}\npub real mean_(vector x) {{\n  return sum(x) / num_elements(x);\n}}\n"),
    );
    with_model(
        &p,
        "library {\n  import stats\n}\n\ndata {\n  vector[3] y;\n}\nmodel {\n  real m = stats::mean_(y);\n}\n",
    );

    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");
    assert!(!stan.contains("begin @use"), "{stan}");
    assert!(!stan.contains("@template"), "{stan}");
    assert!(!stan.contains('$'), "{stan}");
    assert!(stan.contains("real stats__mean_(vector x)"), "{stan}");
    // Deterministic, and `--check` agrees.
    assert!(p
        .run(&["build", "model.laplace", "--check"])
        .status
        .success());
    stanc_accepts(&p, "build/model.stan");
}

// ---------------------------------------------------------------------------
// Patch 1 session 4: statement macros (@macro / @expand)
// ---------------------------------------------------------------------------

const MACROS_LIB: &str = r#"// @laplace
// @brief Give several parameters the same prior.
pub @macro priors(each $p: ident, $dist: expr) : stmt in model {
  $p ~ $dist;
}

// @laplace
// @brief Standardize several parameters by a shared scale.
pub @macro z_scores(each $p: ident, $scale: expr) : stmt in transformed parameters {
  real ${p}_z = $p / half($scale);
}

@macro internal_only($p: ident) : stmt in model {
  $p ~ std_normal();
}

pub @template ncp($name: ident, $N: expr) {
  parameters {
    vector[$N] ${name}_raw;
    real<lower=0> ${name}_sigma;
  }
  transformed parameters {
    vector[$N] $name = ${name}_sigma * ${name}_raw;
  }
  model {
    ${name}_raw ~ std_normal();
  }
}

real half(real x) {
  return x / 2;
}
"#;

fn macros_project() -> Project {
    let p = setup();
    p.write_package_file(
        "stats",
        "1.0.0",
        "laplace.toml",
        "name = \"stats\"\nversion = \"1.0.0\"\n",
    );
    p.write_package_file("stats", "1.0.0", "stats.laplacelib", MACROS_LIB);
    p
}

#[test]
fn a_macro_expands_in_place_once_per_list_element() {
    let p = macros_project();
    with_model(
        &p,
        concat!(
            "library {\n  import stats\n}\n",
            "data {\n  int<lower=1> N;\n  vector[N] y;\n  vector[N] x;\n}\n",
            "parameters {\n  real alpha;\n  real beta;\n  real<lower=0> gamma;\n}\n",
            "model {\n",
            "  @expand stats::priors([alpha, beta, gamma], normal(0, 1));\n",
            "  y ~ normal(alpha + beta * x, gamma);\n",
            "}\n",
        ),
    );

    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");

    assert!(
        stan.contains(concat!(
            "model {\n",
            "  // begin @expand stats::priors -- model.laplace:15\n",
            "  alpha ~ normal(0, 1);\n",
            "  beta ~ normal(0, 1);\n",
            "  gamma ~ normal(0, 1);\n",
            "  // end @expand stats::priors\n",
            "  y ~ normal(alpha + beta * x, gamma);\n",
        )),
        "{stan}"
    );
    // A distribution passed as an `expr` must not be parenthesized.
    assert!(!stan.contains("~ (normal"), "{stan}");
    assert!(
        !stan
            .lines()
            .any(|line| line.trim_start().starts_with("@expand")),
        "{stan}"
    );
    assert!(!stan.contains("@macro"), "{stan}");
    assert!(!stan.contains('$'), "{stan}");
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn a_declaring_macro_builds_one_name_per_element_and_calls_its_own_helper() {
    let p = macros_project();
    with_model(
        &p,
        concat!(
            "library {\n  import stats\n}\n",
            "parameters {\n  real alpha;\n  real beta;\n}\n",
            "transformed parameters {\n",
            "  @expand stats::z_scores([alpha, beta], 4.0);\n",
            "}\n",
            "model {\n}\n",
        ),
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");

    assert!(
        stan.contains("  real alpha_z = alpha / stats__half(4.0);"),
        "{stan}"
    );
    assert!(
        stan.contains("  real beta_z = beta / stats__half(4.0);"),
        "{stan}"
    );
    // The helper is private to `stats` and resolved in its scope.
    assert!(stan.contains("real stats__half(real x)"), "{stan}");
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn templates_and_macros_compose_in_one_model() {
    let p = macros_project();
    with_model(
        &p,
        concat!(
            "library {\n  import stats\n}\n",
            "@use stats::ncp(theta, K);\n",
            "data {\n  int<lower=1> K;\n  vector[K] y;\n}\n",
            "parameters {\n  real alpha;\n  real<lower=0> gamma;\n}\n",
            "transformed parameters {\n",
            "  @expand stats::z_scores([alpha], 4.0);\n",
            "}\n",
            "model {\n",
            "  @expand stats::priors([alpha, gamma], normal(0, 1));\n",
            "  y ~ normal(alpha + theta[1], gamma);\n",
            "}\n",
        ),
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");

    // The template's piece comes first in `transformed parameters`,
    // then the macro expands where it was written.
    let theta = stan
        .find("vector[K] theta = theta_sigma * theta_raw;")
        .unwrap();
    let alpha_z = stan.find("real alpha_z = alpha /").unwrap();
    assert!(theta < alpha_z, "{stan}");
    assert!(stan.contains("alpha ~ normal(0, 1);"), "{stan}");
    stanc_accepts(&p, "build/model.stan");
}

#[test]
fn an_empty_list_is_an_error_rather_than_silently_expanding_to_nothing() {
    let p = macros_project();
    with_model(
        &p,
        "library {\n  import stats\n}\nparameters {\n  real alpha;\n}\nmodel {\n  @expand stats::priors([], normal(0, 1));\n}\n",
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("the list for `$p` is empty"), "{err}");
    assert!(err.contains("would produce nothing"), "{err}");
    assert!(err.contains("at least one element"), "{err}");
}

#[test]
fn expanding_a_macro_in_a_block_it_does_not_declare_is_an_error() {
    let p = macros_project();
    with_model(
        &p,
        concat!(
            "library {\n  import stats\n}\n",
            "parameters {\n  real alpha;\n}\n",
            "generated quantities {\n  @expand stats::priors([alpha], normal(0, 1));\n}\n",
        ),
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains("cannot be expanded in `generated quantities`"),
        "{err}"
    );
    assert!(err.contains("it declares `in model`"), "{err}");
}

#[test]
fn a_macro_whose_body_cannot_go_where_it_claims_is_rejected_at_the_definition() {
    let cases: &[(&str, &str, &str)] = &[
        (
            "a `~` body targeting generated quantities",
            "pub @macro m($p: ident) : stmt in generated quantities {\n  $p ~ std_normal();\n}\n",
            "`~` statement is not legal in `generated quantities`",
        ),
        (
            "an `_rng` body targeting the model block",
            "pub @macro m($p: ident) : stmt in model {\n  real ${p}_s = normal_rng(0, 1);\n}\n",
            "`_rng` function cannot be called in `model`",
        ),
        (
            "two `each` parameters",
            "pub @macro m(each $a: ident, each $b: ident) : stmt in model {\n  $a ~ std_normal();\n}\n",
            "marks 2 parameters `each`",
        ),
        (
            "an unknown target block",
            "pub @macro m($p: ident) : stmt in priors {\n  $p ~ std_normal();\n}\n",
            "is not a Stan program block",
        ),
        (
            "a body reaching for a model variable",
            "pub @macro m($p: ident) : stmt in model {\n  $p ~ normal(mu, 1);\n}\n",
            "which it does not declare",
        ),
        (
            "a fixed-name declaration",
            "pub @macro m($p: ident) : stmt in transformed parameters {\n  real tmp = 1;\n  real ${p}_z = tmp;\n}\n",
            "with a fixed name",
        ),
        (
            "a nested @expand",
            "pub @macro m($p: ident) : stmt in model {\n  @expand other::n($p);\n}\n",
            "nested `@expand`",
        ),
    ];

    for (what, library, expected) in cases {
        let p = setup();
        p.write_package_file(
            "bad",
            "1.0.0",
            "laplace.toml",
            "name = \"bad\"\nversion = \"1.0.0\"\n",
        );
        p.write_package_file("bad", "1.0.0", "bad.laplacelib", library);

        let out = p.run(&["add", "bad"]);
        assert!(!out.status.success(), "{what} should not be accepted");
        let err = stderr(&out);
        assert!(
            err.contains(expected),
            "{what}: expected {expected:?} in\n{err}"
        );
        assert!(
            err.contains("bad.laplacelib:"),
            "{what} needs a location: {err}"
        );
        assert!(err.contains("help: "), "{what} needs a help line: {err}");
    }
}

#[test]
fn a_macro_expansion_that_would_redeclare_a_name_is_refused() {
    let p = macros_project();

    // Against one of the model's own declarations.
    with_model(
        &p,
        concat!(
            "library {\n  import stats\n}\n",
            "parameters {\n  real alpha;\n  real alpha_z;\n}\n",
            "transformed parameters {\n  @expand stats::z_scores([alpha], 4.0);\n}\n",
        ),
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("`alpha_z` is declared twice"),
        "{}",
        stderr(&out)
    );

    // And against itself, when an element is repeated.
    p.write_project_file(
        "model.laplace",
        concat!(
            "library {\n  import stats\n}\n",
            "parameters {\n  real alpha;\n}\n",
            "transformed parameters {\n  @expand stats::z_scores([alpha, alpha], 4.0);\n}\n",
        ),
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("`alpha_z` is declared twice"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn a_private_macro_cannot_be_expanded_from_a_model() {
    let p = macros_project();
    with_model(
        &p,
        "library {\n  import stats\n}\nparameters {\n  real a;\n}\nmodel {\n  @expand stats::internal_only(a);\n}\n",
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("`stats::internal_only` is private"), "{err}");
    assert!(err.contains("only macros marked `pub`"), "{err}");
}

#[test]
fn an_each_parameter_needs_a_list_and_says_so() {
    let p = macros_project();
    with_model(
        &p,
        "library {\n  import stats\n}\nparameters {\n  real a;\n}\nmodel {\n  @expand stats::priors(a, normal(0, 1));\n}\n",
    );
    let out = p.run(&["build", "model.laplace"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains("is an `each` parameter, so it needs a list"),
        "{err}"
    );
    assert!(err.contains("[a, b, c]"), "{err}");
}

#[test]
fn a_project_with_no_expand_statements_is_unaffected_by_macros() {
    let p = setup();
    p.write_package_file(
        "stats",
        "1.0.0",
        "laplace.toml",
        "name = \"stats\"\nversion = \"1.0.0\"\n",
    );
    p.write_package_file(
        "stats",
        "1.0.0",
        "stats.laplacelib",
        &format!(
            "{MACROS_LIB}\npub real mean_(vector x) {{\n  return sum(x) / num_elements(x);\n}}\n"
        ),
    );
    with_model(
        &p,
        "library {\n  import stats\n}\n\ndata {\n  vector[3] y;\n}\nmodel {\n  real m = stats::mean_(y);\n}\n",
    );

    let out = p.run(&["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let stan = p.read_project_file("build/model.stan");
    assert!(!stan.contains("begin @expand"), "{stan}");
    assert!(!stan.contains("@macro"), "{stan}");
    assert!(!stan.contains('$'), "{stan}");
    assert!(stan.contains("real stats__mean_(vector x)"), "{stan}");
    assert!(p
        .run(&["build", "model.laplace", "--check"])
        .status
        .success());
    stanc_accepts(&p, "build/model.stan");
}
