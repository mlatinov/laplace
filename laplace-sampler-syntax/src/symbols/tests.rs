use std::path::Path;

use laplace::manifest::PackageKind;
use laplace::resolve::lockfile::{write_lockfile, LockedPackage, Lockfile};

use super::*;
use crate::editor::check_lapsampler;
use crate::lapsampler::parse::{parse, KExpr};

const MALA: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../laplace-sample/kernels/mala.lapkernel"
);

// ---- builtins ----------------------------------------------------------------

#[test]
fn builtins_are_the_registry_with_terminals_marked() {
    let all = builtins();
    let names: Vec<&str> = all.iter().map(|b| b.name).collect();
    let registry: Vec<&str> = Registry::standard()
        .entries()
        .iter()
        .map(|e| e.name)
        .collect();
    assert_eq!(names, registry);
    let kind = |n: &str| all.iter().find(|b| b.name == n).unwrap().kind;
    assert_eq!(kind("q"), BuiltinKind::Value);
    assert_eq!(kind("dim"), BuiltinKind::Value);
    assert_eq!(kind("log_density_grad"), BuiltinKind::Function);
    assert_eq!(kind("propose"), BuiltinKind::Terminal);
    assert_eq!(kind("move"), BuiltinKind::Terminal);
    let terminals: Vec<&str> = all
        .iter()
        .filter(|b| b.kind == BuiltinKind::Terminal)
        .map(|b| b.name)
        .collect();
    assert_eq!(terminals, ["propose", "move"]);
    for b in &all {
        assert!(!b.signatures.is_empty() && !b.doc.is_empty(), "{}", b.name);
    }
}

// ---- combinators -------------------------------------------------------------

/// The `<name>_call` rules of the grammar: the combinators it parses.
fn grammar_combinators() -> Vec<String> {
    include_str!("../lapsampler/grammar.pest")
        .lines()
        .filter_map(|l| {
            // `seq_call = { kw_seq ~ ...`: a call introduced by its keyword
            // (`named_call`, a built-in kernel, has none).
            let (rule, body) = l.split_once('=')?;
            let name = rule.trim().strip_suffix("_call")?;
            body.trim()
                .starts_with(&format!("{{ kw_{name} "))
                .then(|| name.to_string())
        })
        .collect()
}

#[test]
fn the_combinators_are_the_grammars() {
    let mut ours: Vec<&str> = combinators().iter().map(|c| c.name).collect();
    let mut grammar = grammar_combinators();
    ours.sort();
    grammar.sort();
    assert_eq!(ours, grammar);
}

fn sampler(kernel: &str) -> String {
    format!("sampler s {{\n  kernel {{\n    rest : {kernel}\n  }}\n}}\n")
}

fn empty_project(dir: &Path) -> Project {
    Project {
        dir: dir.to_path_buf(),
        cache_root: dir.join("cache"),
    }
}

#[test]
fn each_example_parses_as_its_combinator_and_checks_where_allowed() {
    let tmp = tempfile::tempdir().unwrap();
    let project = empty_project(tmp.path());
    for c in combinators() {
        assert!(c.example.starts_with(&format!("{}(", c.name)), "{}", c.name);
        assert!(
            c.signature.starts_with(&format!("{}(", c.name)),
            "{}",
            c.name
        );
        let src = sampler(c.example);
        let ast = parse("s.lapsampler", &src).unwrap_or_else(|e| panic!("{}: {e}", c.example));
        let root = match &ast.lines[0].kernel {
            KExpr::Seq(..) => "seq",
            KExpr::Mix(..) => "mix",
            KExpr::Repeat(..) => "repeat",
            KExpr::Adapt(..) => "adapt",
            KExpr::On(..) => "on",
            other => panic!("{other:?}"),
        };
        assert_eq!(root, c.name);
        let ds = check_lapsampler("s.lapsampler", &src, tmp.path(), &project);
        assert_eq!(ds.is_empty(), c.in_kernel_lines, "{}: {ds:?}", c.example);
    }
}

#[test]
fn adapt_takes_exactly_its_listed_settings() {
    let adapt = combinators().iter().find(|c| c.name == "adapt").unwrap();
    assert_eq!(adapt.settings, ["target_accept"]);
    let tmp = tempfile::tempdir().unwrap();
    let ds = check_lapsampler(
        "s.lapsampler",
        &sampler("adapt(mala(), bogus = 1)"),
        tmp.path(),
        &empty_project(tmp.path()),
    );
    assert_eq!(ds.len(), 1);
    assert!(
        ds[0].message.contains("only target_accept"),
        "{}",
        ds[0].message
    );
}

// ---- kernel_index ------------------------------------------------------------

/// A project with a kernel package `langevin` (a path dependency, with
/// mala and a broken kernel), a function library, a local kernel, and a
/// kernel in a hidden directory.
fn project() -> (tempfile::TempDir, Project) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let pkg = root.join("langevin");
    std::fs::create_dir_all(pkg.join("kernels")).unwrap();
    std::fs::write(
        pkg.join("laplace.toml"),
        "name = \"langevin\"\nversion = \"0.1.0\"\nkind = \"kernels\"\nlaplace = \">=0.3\"\n",
    )
    .unwrap();
    std::fs::copy(MALA, pkg.join("kernels/mala.lapkernel")).unwrap();
    std::fs::write(
        pkg.join("kernels/broken.lapkernel"),
        "kernel broken {\n  step {\n    move(nope);\n  }\n}\n",
    )
    .unwrap();
    let dir = root.join("project");
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    std::fs::create_dir_all(dir.join("samplers")).unwrap();
    std::fs::create_dir_all(dir.join(".hidden")).unwrap();
    std::fs::copy(MALA, dir.join("kernels/mine.lapkernel")).unwrap();
    std::fs::copy(MALA, dir.join(".hidden/skipped.lapkernel")).unwrap();
    let lock = Lockfile {
        root: vec!["gps".into(), "langevin".into()],
        packages: vec![
            LockedPackage::leaf("gps", "1.0.0", "sha256:0", "registry"),
            LockedPackage {
                kind: PackageKind::Kernels,
                ..LockedPackage::leaf("langevin", "0.1.0", "sha256:0", "path+../langevin")
            },
        ],
    };
    write_lockfile(&dir.join("laplace.lock"), &lock).unwrap();
    let project = Project {
        dir,
        cache_root: root.join("home/.laplace/packages"),
    };
    (tmp, project)
}

fn names(kernels: &[KernelEntry]) -> Vec<&str> {
    kernels.iter().map(|k| k.name.as_str()).collect()
}

#[test]
fn the_index_lists_built_ins_package_kernels_and_local_files() {
    let (_tmp, p) = project();
    let (kernels, problems) = kernel_index(&p, &p.dir);
    assert_eq!(problems, Vec::<String>::new());
    assert_eq!(
        names(&kernels),
        [
            "nuts",
            "rwm",
            "mala",
            "hmc",
            "langevin::broken",
            "langevin::mala",
            "./kernels/mine.lapkernel"
        ]
    );
}

#[test]
fn local_files_are_named_relative_to_the_sampler() {
    let (_tmp, p) = project();
    let samplers = p.dir.join("samplers");
    let (kernels, _) = kernel_index(&p, &samplers);
    let mine = kernels.last().unwrap();
    assert_eq!(mine.name, "../kernels/mine.lapkernel");
    assert_eq!(mine.origin, KernelOrigin::Local);
    assert_eq!(
        mine.file.as_deref(),
        Some(p.dir.join("kernels/mine.lapkernel").as_path())
    );
    // It is what a .lapsampler there writes, and it checks.
    let src = sampler(&format!("adapt({})", mine.name));
    assert_eq!(
        check_lapsampler("s.lapsampler", &src, &samplers, &p),
        vec![]
    );
}

#[test]
fn a_package_kernel_has_its_tuning_values_terminals_and_doc() {
    let (_tmp, p) = project();
    let (kernels, _) = kernel_index(&p, &p.dir);
    let mala = kernels.iter().find(|k| k.name == "langevin::mala").unwrap();
    assert_eq!(
        mala.origin,
        KernelOrigin::Package {
            package: "langevin".into(),
            version: "0.1.0".into()
        }
    );
    let file = mala.file.as_ref().unwrap();
    assert!(
        file.ends_with("langevin/kernels/mala.lapkernel"),
        "{}",
        file.display()
    );
    assert!(file.is_file());
    assert_eq!(
        mala.brief.as_deref(),
        Some("Metropolis-adjusted Langevin algorithm with a diagonal metric")
    );
    assert_eq!(
        mala.settings,
        vec![
            KernelSetting {
                name: "eps".into(),
                decl: Some("real<lower=0> eps = 1".into()),
                default: Some("1".into()),
                adapt: Some("@adapt(step_size, target_accept = 0.574)".into()),
                doc: Some("Step size (adapted toward 0.574 acceptance)".into()),
            },
            KernelSetting {
                name: "m".into(),
                decl: Some("vector<lower=0>[dim] m = rep_vector(1, dim)".into()),
                default: Some("rep_vector(1, dim)".into()),
                adapt: Some("@adapt(metric)".into()),
                doc: Some("Diagonal inverse metric (adapted in warmup)".into()),
            },
        ]
    );
    assert_eq!(mala.ends_in, ["propose"]);
    assert_eq!(mala.error, None);
    // The same text as `laplace sampler doc langevin::mala`.
    assert_eq!(mala.doc, doc("langevin::mala", &p).unwrap());
}

#[test]
fn a_kernel_that_does_not_check_is_listed_with_its_error() {
    let (_tmp, p) = project();
    let (kernels, _) = kernel_index(&p, &p.dir);
    let broken = kernels
        .iter()
        .find(|k| k.name == "langevin::broken")
        .unwrap();
    assert!(broken.file.as_ref().unwrap().is_file());
    let e = broken.error.as_deref().unwrap();
    assert!(e.contains("unknown name `nope`"), "{e}");
    assert!(broken.settings.is_empty() && broken.doc.is_empty());
}

#[test]
fn a_built_in_has_its_sampler_settings_with_defaults() {
    let (_tmp, p) = project();
    let (kernels, _) = kernel_index(&p, &p.dir);
    for b in BuiltIn::ALL {
        let k = kernels.iter().find(|k| k.name == b.name()).unwrap();
        assert_eq!(k.origin, KernelOrigin::BuiltIn);
        assert_eq!(k.file, None);
        let settings: Vec<&str> = k.settings.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(settings, b.sampler_settings());
        assert!(
            k.settings.iter().all(|s| s.default.is_some()),
            "{}",
            b.name()
        );
        assert_eq!(k.doc, doc(b.name(), &p).unwrap());
    }
    let hmc = kernels.iter().find(|k| k.name == "hmc").unwrap();
    let steps = hmc.settings.iter().find(|s| s.name == "steps").unwrap();
    assert_eq!(steps.default.as_deref(), Some("10"));
}

#[test]
fn every_built_in_setting_has_a_default() {
    for b in BuiltIn::ALL {
        for s in b.sampler_settings() {
            assert!(b.defaults().iter().any(|(k, _)| k == s), "{} {s}", b.name());
        }
    }
}

#[test]
fn without_a_lockfile_there_are_built_ins_and_local_files_only() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::copy(MALA, tmp.path().join("k.lapkernel")).unwrap();
    let (kernels, problems) = kernel_index(&empty_project(tmp.path()), tmp.path());
    assert_eq!(
        names(&kernels),
        ["nuts", "rwm", "mala", "hmc", "./k.lapkernel"]
    );
    assert_eq!(problems, Vec::<String>::new());
}

#[test]
fn without_a_lockfile_only_nearby_files_are_listed() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    for f in [
        "k.lapkernel",
        "kernels/a.lapkernel",
        "kernels/deep/b.lapkernel",
        "far/c.lapkernel",
        "far/kernels/d.lapkernel",
    ] {
        let path = dir.join(f);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::copy(MALA, path).unwrap();
    }
    let local = |p: &Project, from: &Path| -> Vec<String> {
        let (kernels, _) = kernel_index(p, from);
        kernels[BuiltIn::ALL.len()..]
            .iter()
            .map(|k| k.name.clone())
            .collect()
    };
    assert_eq!(
        local(&empty_project(dir), dir),
        [
            "./k.lapkernel",
            "./kernels/a.lapkernel",
            "./kernels/deep/b.lapkernel"
        ]
    );
    // The same tree with a lock is a project: every file is listed.
    std::fs::write(dir.join("laplace.lock"), "").unwrap();
    assert_eq!(
        local(&empty_project(dir), dir),
        [
            "./far/c.lapkernel",
            "./far/kernels/d.lapkernel",
            "./k.lapkernel",
            "./kernels/a.lapkernel",
            "./kernels/deep/b.lapkernel"
        ]
    );
    // A sampler outside the project adds only its own nearby files.
    let outside = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(outside.path().join("x/kernels")).unwrap();
    std::fs::copy(MALA, outside.path().join("x/kernels/e.lapkernel")).unwrap();
    std::fs::copy(MALA, outside.path().join("f.lapkernel")).unwrap();
    let from = outside.path().join("x");
    let names = local(&empty_project(dir), &from);
    assert!(
        names.contains(&"./kernels/e.lapkernel".to_string()),
        "{names:?}"
    );
    assert!(
        !names.iter().any(|n| n.ends_with("f.lapkernel")),
        "{names:?}"
    );
}

#[test]
fn a_package_that_does_not_resolve_is_a_problem_not_an_entry() {
    let (_tmp, p) = project();
    std::fs::remove_dir_all(p.dir.join("../langevin")).unwrap();
    let (kernels, problems) = kernel_index(&p, &p.dir);
    assert!(!names(&kernels).iter().any(|n| n.starts_with("langevin::")));
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("langevin"), "{}", problems[0]);
}

#[test]
fn relative_names() {
    let r = |from: &str, file: &str| relative_name(Path::new(from), Path::new(file));
    assert_eq!(r("/p", "/p/k.lapkernel"), "./k.lapkernel");
    assert_eq!(r("/p", "/p/kernels/k.lapkernel"), "./kernels/k.lapkernel");
    assert_eq!(
        r("/p/samplers", "/p/kernels/k.lapkernel"),
        "../kernels/k.lapkernel"
    );
    assert_eq!(r("/p/a/b", "/p/k.lapkernel"), "../../k.lapkernel");
}

#[test]
fn a_built_ins_source_is_its_implementation() {
    for b in BuiltIn::ALL {
        let (path, source) = b.source();
        assert!(path.ends_with(&format!("{}.rs", b.name())), "{path}");
        let on_disk =
            std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(path))
                .unwrap();
        assert_eq!(source, on_disk);
    }
}

/// Every `@adapt` tag the editor documents is one the checker accepts, and
/// the other way round.
#[test]
fn adapt_tags_are_the_checkers() {
    let names: Vec<_> = adapt_tags().iter().map(|t| t.name).collect();
    assert_eq!(names, ["step_size", "metric"]);
    for t in adapt_tags() {
        let src = format!(
            "kernel k {{\n  tuning {{\n    {}\n  }}\n  step {{\n    propose(q, 0);\n  }}\n}}\n",
            t.signature
        );
        let ds = crate::editor::check_lapkernel("k.lapkernel", &src, None);
        assert!(ds.is_empty(), "{}: {ds:?}", t.name);
    }
}
