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
        let pkg_dir = self.registry_dir().join(name).join(version);
        fs::create_dir_all(&pkg_dir).unwrap();
        let exports_toml = exports
            .iter()
            .map(|e| format!("\"{e}\""))
            .collect::<Vec<_>>()
            .join(", ");
        fs::write(
            pkg_dir.join("laplace.toml"),
            format!("name = \"{name}\"\nversion = \"{version}\"\nexports = [{exports_toml}]\n"),
        )
        .unwrap();
        fs::write(pkg_dir.join(format!("{name}.stan")), stan_body).unwrap();
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
