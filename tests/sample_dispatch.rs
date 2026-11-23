//! `laplace sample` / `sampler` / `new` are forwarded to the separate
//! `laplace-sample` binary found on PATH (D1). These tests put a fake
//! `laplace-sample` script on a controlled PATH, so they need neither the
//! real sampler nor BridgeStan.
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, Output};

fn laplace(path_env: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_laplace"))
        .args(args)
        .env("PATH", path_env)
        .output()
        .unwrap()
}

/// A `laplace-sample` that prints its arguments and exits with 7.
fn fake_sampler(dir: &Path) {
    let p = dir.join("laplace-sample");
    fs::write(&p, "#!/bin/sh\necho \"args:$*\"\nexit 7\n").unwrap();
    fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn sampler_subcommands_forward_arguments_and_exit_status() {
    let tmp = tempfile::tempdir().unwrap();
    fake_sampler(tmp.path());
    for args in [
        vec!["sample", "--help"],
        vec![
            "sample",
            "model.laplace",
            "--sampler",
            "mala",
            "--set",
            "step_size=0.1",
        ],
        vec!["sampler", "check", "hybrid.lapsampler"],
        vec!["new", "kernel", "mala"],
    ] {
        let out = laplace(tmp.path(), &args);
        assert_eq!(out.status.code(), Some(7), "{args:?}");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            format!("args:{}", args.join(" "))
        );
    }
}

#[test]
fn missing_sampler_binary_prints_an_install_hint() {
    let tmp = tempfile::tempdir().unwrap(); // empty PATH directory
    let out = laplace(tmp.path(), &["sample", "model.laplace"]);
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("`laplace sample` needs the `laplace-sample` binary"),
        "{err}"
    );
    assert!(err.contains("cargo install --path laplace-sample"), "{err}");
}

#[test]
fn other_unknown_subcommands_are_still_rejected_by_clap() {
    let tmp = tempfile::tempdir().unwrap();
    fake_sampler(tmp.path());
    let out = laplace(tmp.path(), &["frobnicate"]);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("unrecognized subcommand 'frobnicate'"),
        "{err}"
    );
    assert!(!String::from_utf8_lossy(&out.stdout).contains("args:"));
}

/// Phase 4.3: `laplace doc pkg::kernel` for a kernel package (laplace.lock
/// says `kind = "kernels"`) is answered by laplace-sample; a function
/// library's doc is not forwarded.
#[test]
fn doc_of_a_kernel_package_is_forwarded() {
    let tmp = tempfile::tempdir().unwrap();
    let bin = tmp.path().join("bin");
    fs::create_dir(&bin).unwrap();
    fake_sampler(&bin);
    let project = tmp.path().join("project");
    fs::create_dir(&project).unwrap();
    fs::write(
        project.join("laplace.lock"),
        "root = [\"gps\", \"langevin\"]\n\n[[package]]\nname = \"gps\"\nversion = \"1.0.0\"\n\
         checksum = \"sha256:0\"\nsource = \"registry\"\ndependencies = []\n\n[[package]]\n\
         name = \"langevin\"\nversion = \"0.1.0\"\nchecksum = \"sha256:0\"\nsource = \"registry\"\n\
         dependencies = []\nkind = \"kernels\"\n",
    )
    .unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_laplace"))
            .args(args)
            .current_dir(&project)
            .env("PATH", &bin)
            .env("HOME", tmp.path())
            .output()
            .unwrap()
    };
    let out = run(&["doc", "langevin::mala"]);
    assert_eq!(out.status.code(), Some(7));
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "args:sampler doc langevin::mala"
    );
    let out = run(&["doc", "langevin::mala", "--html", "-o", "x.html"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--html is for function libraries"));
    // A function library is looked up by the compiler itself.
    let out = run(&["doc", "gps::rbf"]);
    assert!(!String::from_utf8_lossy(&out.stdout).contains("args:"));
}
