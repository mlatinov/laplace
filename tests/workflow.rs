//! End-to-end tests for the package-author and compiler-maintenance
//! workflow: `--version`, the `laplace` manifest key, `init --update`, cache
//! refresh, path dependencies, `release` and `self-update`.
//!
//! Like `tests/cli.rs`, every test runs the real binary against temp
//! directories with `HOME` pointed at a fake home. Git remotes are local bare
//! repositories; nothing here touches the network.

// TODO(phase 7): drop once every helper has a caller.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Env {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    home: PathBuf,
    extra_env: Vec<(String, String)>,
}

fn setup() -> Env {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let home = root.join("home");
    fs::create_dir_all(home.join(".laplace").join("registry")).unwrap();
    Env {
        _tmp: tmp,
        root,
        home,
        extra_env: Vec::new(),
    }
}

impl Env {
    /// A fresh directory under the test root.
    fn dir(&self, name: &str) -> PathBuf {
        let dir = self.root.join(name);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn registry(&self) -> PathBuf {
        self.home.join(".laplace").join("registry")
    }

    fn cache(&self) -> PathBuf {
        self.home.join(".laplace").join("packages")
    }

    fn run_in(&self, cwd: &Path, args: &[&str]) -> Output {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_laplace"));
        cmd.args(args)
            .current_dir(cwd)
            .env("HOME", &self.home)
            .env_remove("LAPLACE_REGISTRY")
            .env_remove("LAPLACE_RELEASES_URL")
            // Keep git from reading the developer's own config (signing,
            // hooks, default branch) inside `laplace release` tests.
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_AUTHOR_NAME", "Test")
            .env("GIT_AUTHOR_EMAIL", "test@example.com")
            .env("GIT_COMMITTER_NAME", "Test")
            .env("GIT_COMMITTER_EMAIL", "test@example.com");
        for (key, value) in &self.extra_env {
            cmd.env(key, value);
        }
        cmd.output().expect("failed to run laplace binary")
    }
}

fn write(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, contents).unwrap();
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

/// Both streams, for assertions that should not care which one a note went to.
fn all_output(output: &Output) -> String {
    format!("{}{}", stdout(output), stderr(output))
}

// -- Phase 1: compiler versioning -------------------------------------------

#[test]
fn version_flag_prints_the_cargo_version() {
    let env = setup();
    let out = env.run_in(&env.root, &["--version"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let text = stdout(&out);
    assert!(
        text.starts_with(&format!("laplace {}", env!("CARGO_PKG_VERSION"))),
        "{text}"
    );
}

#[test]
fn version_verbose_names_commit_target_and_directories() {
    let env = setup();
    let out = env.run_in(&env.root, &["version", "--verbose"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let text = stdout(&out);
    for key in [
        "version:",
        "commit:",
        "built:",
        "target:",
        "cache:",
        "registry:",
    ] {
        assert!(text.contains(key), "missing {key}: {text}");
    }
    // The directories reported are the ones actually in use.
    assert!(text.contains(&env.cache().display().to_string()), "{text}");
    assert!(
        text.contains(&env.registry().display().to_string()),
        "{text}"
    );
}

// -- Phase 2: minimum compiler version in manifests -------------------------

const RBF_STAN: &str = "// @laplace\n// @brief RBF covariance.\nmatrix rbf_cov(vector x, real alpha, real rho) {\n  return gp_exp_quad_cov(x, alpha, rho);\n}\n";

const MODEL: &str = "library {\n  import gps\n}\n\ndata {\n  int N;\n  vector[N] x;\n}\nmodel {\n  matrix[N, N] K = gps::rbf_cov(x, 1.0, 1.0);\n}\n";

/// A registry package `gps@<version>` exporting `rbf_cov`, with optional
/// extra manifest lines (e.g. a `laplace = ...` requirement).
fn registry_gps(env: &Env, version: &str, extra_manifest: &str) -> PathBuf {
    let dir = env.registry().join("gps").join(version);
    write(
        &dir.join("laplace.toml"),
        &format!(
            "name = \"gps\"\nversion = \"{version}\"\n{extra_manifest}exports = [\"rbf_cov\"]\n"
        ),
    );
    write(&dir.join("gps.stan"), RBF_STAN);
    dir
}

#[test]
fn a_package_requiring_a_newer_compiler_is_refused_with_the_update_hint() {
    let env = setup();
    registry_gps(&env, "1.0.0", "laplace = \">=99.0\"\n");
    let project = env.dir("project");

    let out = env.run_in(&project, &["add", "gps"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("requires laplace >=99.0"), "{err}");
    assert!(
        err.contains(&format!("this is laplace {}", env!("CARGO_PKG_VERSION"))),
        "{err}"
    );
    assert!(err.contains("laplace self-update"), "{err}");
}

#[test]
fn a_satisfied_package_requirement_builds() {
    let env = setup();
    registry_gps(&env, "1.0.0", "laplace = \">=0.2\"\n");
    let project = env.dir("project");
    write(&project.join("model.laplace"), MODEL);

    let out = env.run_in(&project, &["add", "gps"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let out = env.run_in(&project, &["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let out = env.run_in(&project, &["doc", "gps::rbf_cov"]);
    assert!(out.status.success(), "{}", stderr(&out));
}

#[test]
fn a_project_requiring_a_newer_compiler_refuses_build_install_and_doc() {
    let env = setup();
    registry_gps(&env, "1.0.0", "");
    let project = env.dir("project");
    write(&project.join("model.laplace"), MODEL);
    assert!(env.run_in(&project, &["add", "gps"]).status.success());

    // Raise the bar after the fact: the gate applies to every command,
    // including the lock-only ones.
    let manifest = read(&project.join("laplace.toml"));
    write(
        &project.join("laplace.toml"),
        &format!("laplace = \">=99\"\n{manifest}"),
    );
    for args in [
        &["build", "model.laplace"][..],
        &["install"][..],
        &["doc", "gps::rbf_cov"][..],
        &["add", "gps"][..],
    ] {
        let out = env.run_in(&project, args);
        assert!(!out.status.success(), "{args:?} should fail");
        assert!(
            stderr(&out).contains("requires laplace >=99"),
            "{args:?}: {}",
            stderr(&out)
        );
    }
}

#[test]
fn a_manifest_without_the_laplace_key_imposes_nothing() {
    let env = setup();
    registry_gps(&env, "1.0.0", "");
    let project = env.dir("project");
    write(&project.join("model.laplace"), MODEL);
    assert!(env.run_in(&project, &["add", "gps"]).status.success());
    assert!(!read(&project.join("laplace.toml")).contains("laplace ="));
    let out = env.run_in(&project, &["build", "model.laplace"]);
    assert!(out.status.success(), "{}", stderr(&out));
}
