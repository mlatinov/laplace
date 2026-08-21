mod codegen;
mod docs;
mod init;
mod manifest;
mod parser;
mod resolve;
mod validate;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use codegen::InstalledPackage;
use parser::library_block::{parse_library_block, ImportStatement};
use resolve::Registry;

#[derive(Parser)]
#[command(
    name = "laplace",
    about = "Source-to-source preprocessor for Stan: package manager + namespaces + doc lookup"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scan the current directory's .stan files and generate a starter
    /// laplace.toml, guessing `exports` from @laplace-documented functions
    Init,
    /// Compile a .laplace file to .stan
    Build {
        file: PathBuf,
        /// Output path (default: build/<name>.stan)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Diff against the existing build output instead of writing; exits
        /// non-zero if they differ.
        #[arg(long)]
        check: bool,
        /// After writing, type-check the output with `stanc` (must be on
        /// PATH). Off by default so build never requires stanc installed.
        #[arg(long)]
        validate: bool,
    },
    /// Install every package pinned in laplace.lock
    Install,
    /// Add a dependency (optionally pinned to `<pkg>@<version>`), resolving
    /// it and updating laplace.toml + laplace.lock. Pass `--git <url>` with
    /// `--tag <tag>` or `--rev <rev>` to add a git dependency instead of
    /// resolving from the local registry.
    Add {
        package: String,
        #[arg(long)]
        git: Option<String>,
        #[arg(long)]
        tag: Option<String>,
        #[arg(long)]
        rev: Option<String>,
    },
    /// Re-resolve a dependency to the latest version matching its existing
    /// range in laplace.toml
    Update { package: String },
    /// Print a package function's signature and doc comment
    Doc {
        /// `<package>::<function>`
        spec: String,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), CliError> {
    match Cli::parse().command {
        Command::Init => cmd_init(),
        Command::Build {
            file,
            output,
            check,
            validate,
        } => cmd_build(&file, output, check, validate),
        Command::Install => cmd_install(),
        Command::Add {
            package,
            git,
            tag,
            rev,
        } => cmd_add(&package, git.as_deref(), tag.as_deref(), rev.as_deref()),
        Command::Update { package } => cmd_update(&package),
        Command::Doc { spec } => cmd_doc(&spec),
    }
}

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Init(#[from] init::InitError),
    #[error(transparent)]
    LibraryBlock(#[from] parser::library_block::LibraryBlockError),
    #[error(transparent)]
    Codegen(#[from] codegen::CodegenError),
    #[error(transparent)]
    Resolve(Box<resolve::ResolveError>),
    #[error(transparent)]
    Manifest(#[from] manifest::ManifestError),
    #[error(transparent)]
    Lockfile(#[from] resolve::lockfile::LockfileError),
    #[error(transparent)]
    Docs(Box<docs::DocsError>),
    #[error(transparent)]
    Validate(Box<validate::ValidateError>),
    #[error("{0}")]
    Message(String),
    #[error(
        "{} is out of date with `laplace build` (run without --check to update it)",
        .0.display()
    )]
    CheckFailed(PathBuf),
}

// ResolveError is boxed above (clippy::result_large_err) -- write the
// conversion by hand so `?` still works directly on a plain ResolveError.
impl From<resolve::ResolveError> for CliError {
    fn from(err: resolve::ResolveError) -> Self {
        CliError::Resolve(Box::new(err))
    }
}

impl From<docs::DocsError> for CliError {
    fn from(err: docs::DocsError) -> Self {
        CliError::Docs(Box::new(err))
    }
}

impl From<validate::ValidateError> for CliError {
    fn from(err: validate::ValidateError) -> Self {
        CliError::Validate(Box::new(err))
    }
}

fn cmd_init() -> Result<(), CliError> {
    let dir = env::current_dir()?;
    let summary = init::init(&dir)?;

    println!("wrote laplace.toml for `{}`", summary.name);
    if summary.included.is_empty() {
        println!("no @laplace-documented functions found -- exports is empty, add entries by hand");
    } else {
        println!(
            "included {} exported {}: {}",
            summary.included.len(),
            pluralize(summary.included.len(), "function", "functions"),
            summary.included.join(", "),
        );
    }
    if !summary.excluded.is_empty() {
        println!(
            "note: {} undocumented {} left out of exports (add manually if this guess is wrong): {}",
            summary.excluded.len(),
            pluralize(summary.excluded.len(), "function", "functions"),
            summary.excluded.join(", "),
        );
    }

    Ok(())
}

fn cmd_build(
    file: &Path,
    output: Option<PathBuf>,
    check: bool,
    validate: bool,
) -> Result<(), CliError> {
    let source = fs::read_to_string(file)?;
    let library_block = parse_library_block(&source)?;
    let imports: &[ImportStatement] = library_block
        .as_ref()
        .map(|b| b.imports.as_slice())
        .unwrap_or(&[]);

    let lockfile_path = PathBuf::from("laplace.lock");
    let cache_root = default_cache_root();
    let lock = resolve::lockfile::read_lockfile(&lockfile_path)?;

    let mut installed = Vec::with_capacity(imports.len());
    for import in imports {
        let Some(locked) = lock.packages.iter().find(|p| p.name == import.name) else {
            return Err(CliError::Message(format!(
                "`{}` is imported but not recorded in laplace.lock -- run `laplace add {}` first",
                import.name, import.name
            )));
        };
        let package_dir = cache_root.join(&locked.name).join(&locked.version);
        if !package_dir.is_dir() {
            return Err(CliError::Message(format!(
                "`{}@{}` is in laplace.lock but not installed -- run `laplace install` first",
                locked.name, locked.version
            )));
        }
        installed.push(load_installed_package(&package_dir, &locked.name)?);
    }

    let generated = codegen::generate_with_package_lines(&source, library_block.as_ref(), &installed)?;
    let output_path = output.unwrap_or_else(|| default_output_path(file));

    if check {
        let existing = fs::read_to_string(&output_path).unwrap_or_default();
        if existing != generated.source {
            return Err(CliError::CheckFailed(output_path));
        }
        println!("{} is up to date", output_path.display());
        return Ok(());
    }

    if let Some(parent) = output_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, &generated.source)?;

    let lines = generated.source.lines().count();
    println!(
        "wrote {} ({} {}, {} {})",
        output_path.display(),
        lines,
        pluralize(lines, "line", "lines"),
        imports.len(),
        pluralize(imports.len(), "dependency", "dependencies"),
    );

    if validate {
        validate::validate(&stanc_command(), &output_path, &generated)?;
        println!("stanc: OK");
    }

    Ok(())
}

/// Which `stanc` binary/command to invoke for `--validate`. Overridable via
/// `LAPLACE_STANC` so tests never depend on a real stanc install.
fn stanc_command() -> String {
    env::var("LAPLACE_STANC").unwrap_or_else(|_| "stanc".to_string())
}

fn load_installed_package(package_dir: &Path, name: &str) -> Result<InstalledPackage, CliError> {
    let pkg_manifest = manifest::read_package_manifest(&package_dir.join("laplace.toml"))?;
    let source = manifest::read_package_stan_source(package_dir)?;
    let signatures = parser::signatures::extract_signatures(&source);
    Ok(InstalledPackage {
        name: name.to_string(),
        source,
        signatures,
        exported: pkg_manifest.exports,
    })
}

fn default_output_path(file: &Path) -> PathBuf {
    let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    PathBuf::from("build").join(format!("{stem}.stan"))
}

fn pluralize(count: usize, singular: &'static str, plural: &'static str) -> &'static str {
    if count == 1 {
        singular
    } else {
        plural
    }
}

fn cmd_install() -> Result<(), CliError> {
    let lockfile_path = PathBuf::from("laplace.lock");
    let registry = Registry::new(registry_root());
    let cache_root = default_cache_root();

    let installed = resolve::install(&lockfile_path, &registry, &cache_root)?;
    println!(
        "installed {} {}",
        installed.len(),
        pluralize(installed.len(), "package", "packages"),
    );
    Ok(())
}

fn cmd_add(
    spec: &str,
    git: Option<&str>,
    tag: Option<&str>,
    rev: Option<&str>,
) -> Result<(), CliError> {
    let project_manifest_path = PathBuf::from("laplace.toml");
    let lockfile_path = PathBuf::from("laplace.lock");
    let cache_root = default_cache_root();

    if let Some(url) = git {
        let locked = resolve::add_git(
            &project_manifest_path,
            &lockfile_path,
            &cache_root,
            spec,
            url,
            tag,
            rev,
        )?;
        println!("added {}@{} (git)", locked.name, locked.version);
        return Ok(());
    }

    if tag.is_some() || rev.is_some() {
        return Err(CliError::Message(
            "--tag/--rev only apply together with --git".to_string(),
        ));
    }

    let (name, version) = parse_package_spec(spec);
    let registry = Registry::new(registry_root());
    let locked = resolve::add(
        &project_manifest_path,
        &lockfile_path,
        &registry,
        &cache_root,
        name,
        version,
    )?;
    println!("added {}@{}", locked.name, locked.version);
    Ok(())
}

fn cmd_update(package: &str) -> Result<(), CliError> {
    let project_manifest_path = PathBuf::from("laplace.toml");
    let lockfile_path = PathBuf::from("laplace.lock");
    let registry = Registry::new(registry_root());
    let cache_root = default_cache_root();

    let locked = resolve::update(
        &project_manifest_path,
        &lockfile_path,
        &registry,
        &cache_root,
        package,
    )?;
    println!("updated {} to {}", locked.name, locked.version);
    Ok(())
}

fn cmd_doc(spec: &str) -> Result<(), CliError> {
    let Some((package, func)) = spec.split_once("::") else {
        return Err(CliError::Message(format!(
            "expected `<package>::<function>`, got `{spec}`"
        )));
    };

    let lockfile_path = PathBuf::from("laplace.lock");
    let cache_root = default_cache_root();

    let sig = docs::lookup(&lockfile_path, &cache_root, package, func)?;
    print!("{}", docs::render(package, &sig));
    Ok(())
}

/// Split `<pkg>` or `<pkg>@<version>` into its parts.
fn parse_package_spec(spec: &str) -> (&str, Option<&str>) {
    match spec.split_once('@') {
        Some((name, version)) => (name, Some(version)),
        None => (spec, None),
    }
}

fn laplace_home() -> PathBuf {
    let home = env::var_os("HOME").expect("HOME environment variable must be set");
    PathBuf::from(home).join(".laplace")
}

/// Where `laplace install`/`add`/`update` copy resolved packages, per the
/// design doc: `~/.laplace/packages/<name>/<version>/`.
fn default_cache_root() -> PathBuf {
    laplace_home().join("packages")
}

/// The local filesystem "registry" packages are resolved from (real
/// git/http fetching is a later task). Defaults to `~/.laplace/registry`,
/// overridable with `LAPLACE_REGISTRY` -- mainly so tests don't have to
/// touch the real home directory.
fn registry_root() -> PathBuf {
    env::var_os("LAPLACE_REGISTRY")
        .map(PathBuf::from)
        .unwrap_or_else(|| laplace_home().join("registry"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bare_and_pinned_package_specs() {
        assert_eq!(parse_package_spec("gps"), ("gps", None));
        assert_eq!(
            parse_package_spec("gps@1.0.0"),
            ("gps", Some("1.0.0"))
        );
    }

    #[test]
    fn default_output_path_uses_the_input_file_stem() {
        assert_eq!(
            default_output_path(Path::new("model.laplace")),
            PathBuf::from("build/model.stan")
        );
        assert_eq!(
            default_output_path(Path::new("models/gp.laplace")),
            PathBuf::from("build/gp.stan")
        );
    }

    #[test]
    fn pluralize_picks_singular_only_for_exactly_one() {
        assert_eq!(pluralize(0, "line", "lines"), "lines");
        assert_eq!(pluralize(1, "line", "lines"), "line");
        assert_eq!(pluralize(2, "line", "lines"), "lines");
    }
}
