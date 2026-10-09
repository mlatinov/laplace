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

// -- Phase 3: `laplace init` fixes -------------------------------------------

#[test]
fn init_on_a_laplacelib_only_package_has_no_misleading_doc_warning() {
    let env = setup();
    let pkg = env.dir("splines");
    write(
        &pkg.join("splines.laplacelib"),
        "// @laplace\n// @brief Knots.\npub vector knots(int k) {\n  return rep_vector(0, k);\n}\n\nreal helper() {\n  return 1;\n}\n",
    );
    let out = env.run_in(&pkg, &["init"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let text = all_output(&out);
    assert!(!text.contains("doc comment -- exports is empty"), "{text}");
    assert!(!text.contains("none of the"), "{text}");
    assert!(text.contains("public (`pub`, documented): knots"), "{text}");
    assert!(text.contains("private (no `pub`"), "{text}");
    assert!(text.contains("helper"), "{text}");
    assert!(!read(&pkg.join("laplace.toml")).contains("exports"));
}

#[test]
fn init_still_warns_when_stan_sources_lack_doc_comments() {
    let env = setup();
    let pkg = env.dir("gps");
    write(
        &pkg.join("gps.stan"),
        "real jitter(real e) {\n  return e;\n}\n",
    );
    let out = env.run_in(&pkg, &["init"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stdout(&out).contains("has a `// @laplace` doc comment"),
        "{}",
        stdout(&out)
    );
}

#[test]
fn init_on_an_existing_manifest_points_at_update() {
    let env = setup();
    let pkg = env.dir("gps");
    write(
        &pkg.join("laplace.toml"),
        "name = \"gps\"\nversion = \"9.9.9\"\n",
    );
    let out = env.run_in(&pkg, &["init"]);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("already exists -- run `laplace init --update` to sync it"),
        "{}",
        stderr(&out)
    );
    assert!(read(&pkg.join("laplace.toml")).contains("9.9.9"));
}

#[test]
fn init_update_adds_new_exports_once_and_warns_about_stale_ones() {
    let env = setup();
    let pkg = env.dir("gps");
    write(&pkg.join("gps.stan"), RBF_STAN);
    write(
        &pkg.join("laplace.toml"),
        "name = \"gps_custom\"\nversion = \"2.0.0\"\nexports = [\"removed_fn\"]\n",
    );

    let out = env.run_in(&pkg, &["init", "--update"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stdout(&out).contains("rbf_cov"), "{}", stdout(&out));
    assert!(stderr(&out).contains("`removed_fn`"), "{}", stderr(&out));
    assert!(stderr(&out).contains("--prune"), "{}", stderr(&out));
    let manifest = read(&pkg.join("laplace.toml"));
    assert!(
        manifest.contains("gps_custom") && manifest.contains("2.0.0"),
        "{manifest}"
    );

    let again = env.run_in(&pkg, &["init", "--update"]);
    assert!(stdout(&again).contains("up to date"), "{}", stdout(&again));
    assert_eq!(read(&pkg.join("laplace.toml")), manifest);

    let pruned = env.run_in(&pkg, &["init", "--update", "--prune"]);
    assert!(pruned.status.success(), "{}", stderr(&pruned));
    assert!(!read(&pkg.join("laplace.toml")).contains("removed_fn"));
}

// -- git fixtures -------------------------------------------------------------

/// Run git with a clean, deterministic identity and no user config.
fn git(cwd: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_AUTHOR_NAME", "Test")
        .env("GIT_AUTHOR_EMAIL", "test@example.com")
        .env("GIT_COMMITTER_NAME", "Test")
        .env("GIT_COMMITTER_EMAIL", "test@example.com")
        .output()
        .expect("failed to spawn git");
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// A working clone at `<root>/<name>-work` pushing to a bare repository at
/// `<root>/<name>.git`, with one commit on `main` holding `files`.
/// Returns `(work, bare)`.
fn git_repo(env: &Env, name: &str, files: &[(&str, &str)]) -> (PathBuf, PathBuf) {
    let bare = env.root.join(format!("{name}.git"));
    let work = env.root.join(format!("{name}-work"));
    git(
        &env.root,
        &[
            "init",
            "--quiet",
            "--bare",
            "-b",
            "main",
            bare.to_str().unwrap(),
        ],
    );
    git(
        &env.root,
        &["init", "--quiet", "-b", "main", work.to_str().unwrap()],
    );
    for (path, contents) in files {
        write(&work.join(path), contents);
    }
    git(&work, &["add", "."]);
    git(&work, &["commit", "--quiet", "-m", "init"]);
    git(&work, &["remote", "add", "origin", bare.to_str().unwrap()]);
    git(&work, &["push", "--quiet", "-u", "origin", "main"]);
    (work, bare)
}

fn gps_manifest(version: &str) -> String {
    format!("name = \"gps\"\nversion = \"{version}\"\nexports = [\"rbf_cov\"]\n")
}

// -- Phase 4: stale cache and better errors ---------------------------------

#[test]
fn update_refreshes_a_cache_whose_source_changed_without_a_version_bump() {
    let env = setup();
    let pkg = registry_gps(&env, "1.0.0", "");
    let project = env.dir("project");
    assert!(env.run_in(&project, &["add", "gps"]).status.success());
    let cached = env.cache().join("gps").join("1.0.0");
    assert!(read(&cached.join("docs.json")).contains("RBF covariance."));

    // The author edits the package in place, same version.
    write(
        &pkg.join("gps.stan"),
        &RBF_STAN.replace("RBF covariance.", "Squared exponential kernel."),
    );

    // Re-running without changes says nothing about refreshing...
    let out = env.run_in(&project, &["update", "gps"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stdout(&out).contains("refreshed gps@1.0.0 (source changed)"),
        "{}",
        stdout(&out)
    );
    assert!(read(&cached.join("gps.stan")).contains("Squared exponential kernel."));
    assert!(read(&cached.join("docs.json")).contains("Squared exponential kernel."));

    // ...and a second update has nothing left to refresh.
    let again = env.run_in(&project, &["update", "gps"]);
    assert!(!stdout(&again).contains("refreshed"), "{}", stdout(&again));

    // The lock now records the new contents, so install agrees with it.
    let out = env.run_in(&project, &["install"]);
    assert!(out.status.success(), "{}", stderr(&out));
}

#[test]
fn install_refuses_a_source_that_no_longer_matches_the_lock() {
    let env = setup();
    let pkg = registry_gps(&env, "1.0.0", "");
    let project = env.dir("project");
    assert!(env.run_in(&project, &["add", "gps"]).status.success());
    let lock_before = read(&project.join("laplace.lock"));
    let cached = env.cache().join("gps").join("1.0.0").join("gps.stan");
    let cached_before = read(&cached);

    write(&pkg.join("gps.stan"), &RBF_STAN.replace("RBF", "Changed"));

    let out = env.run_in(&project, &["install"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("checksum mismatch for `gps@1.0.0`"), "{err}");
    assert!(err.contains("without a version bump"), "{err}");
    assert!(err.contains("laplace update gps"), "{err}");
    // Neither copy was used: the cache and the lock are untouched.
    assert_eq!(read(&cached), cached_before);
    assert_eq!(read(&project.join("laplace.lock")), lock_before);
}

#[test]
fn a_missing_tag_lists_the_available_tags_and_suggests_the_newest() {
    let env = setup();
    let (work, bare) = git_repo(
        &env,
        "gps",
        &[
            ("laplace.toml", &gps_manifest("0.1.0")),
            ("gps.stan", RBF_STAN),
        ],
    );
    for tag in ["0.1.0", "0.2.0", "0.10.0"] {
        git(&work, &["tag", tag]);
    }
    git(&work, &["push", "--quiet", "origin", "--tags"]);
    let project = env.dir("project");

    let out = env.run_in(
        &project,
        &[
            "add",
            "gps",
            "--git",
            bare.to_str().unwrap(),
            "--tag",
            "0.3.0",
        ],
    );
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("`0.3.0` does not exist"), "{err}");
    assert!(
        err.contains("available tags: 0.1.0, 0.10.0, 0.2.0"),
        "{err}"
    );
    assert!(err.contains("did you mean `--tag 0.10.0`?"), "{err}");
    assert!(!err.contains("Remote branch"), "{err}");
}

#[test]
fn a_tag_naming_a_different_version_than_the_manifest_warns() {
    let env = setup();
    let (work, bare) = git_repo(
        &env,
        "gps",
        &[
            ("laplace.toml", &gps_manifest("0.1.1")),
            ("gps.stan", RBF_STAN),
        ],
    );
    git(&work, &["tag", "0.1.2"]);
    git(&work, &["push", "--quiet", "origin", "0.1.2"]);
    let project = env.dir("project");

    let out = env.run_in(
        &project,
        &[
            "add",
            "gps",
            "--git",
            bare.to_str().unwrap(),
            "--tag",
            "0.1.2",
        ],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("warning: tag 0.1.2 of `gps` contains version 0.1.1 in laplace.toml"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn doc_on_a_private_item_names_the_installed_version_and_what_to_check() {
    let env = setup();
    let dir = env.registry().join("stats").join("1.0.0");
    write(
        &dir.join("laplace.toml"),
        "name = \"stats\"\nversion = \"1.0.0\"\n",
    );
    write(
        &dir.join("stats.laplacelib"),
        "// @laplace\n// @brief Mean.\nreal mean_(vector x) {\n  return mean(x);\n}\n",
    );
    let project = env.dir("project");
    assert!(env.run_in(&project, &["add", "stats"]).status.success());

    let out = env.run_in(&project, &["doc", "stats::mean_"]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains("installed `stats@1.0.0` defines `mean_` without `pub`"),
        "{err}"
    );
    assert!(err.contains("laplace update stats"), "{err}");
}
