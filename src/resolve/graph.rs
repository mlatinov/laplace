//! Dependency-graph resolution: turn a set of root requirements plus a
//! package source into one concrete version per package name, or a clear
//! conflict/cycle error.
//!
//! This module is deliberately free of filesystem, git, and parsing
//! concerns -- everything it needs about the outside world arrives through
//! the [`PackageProvider`] trait, so the algorithm can be unit-tested
//! against synthetic graphs (see the tests at the bottom of this file).
//!
//! # Locked design decisions
//!
//! **The dependency graph is a DAG, not a flat list.** A package declares
//! its own dependencies in its `laplace.toml`, so `regression` can depend
//! on `stats` without the top-level project mentioning `stats` at all.
//!
//! **Diamond dependencies unify to exactly one version (Cargo-style).**
//! Every requirement on a package name -- from the root project and from
//! every other package -- is collected, and one version satisfying *all* of
//! them is chosen (the highest available). If no such version exists, that
//! is a hard error naming every requirer and its range. Two versions of the
//! same package never coexist in one build: they would mangle to the same
//! `pkg__func` names and silently clobber each other in the generated Stan.
//!
//! **Cycles are a hard error, not a silently-broken build.** `A` importing
//! `B` importing `A` is reported with the full cycle path.

use std::collections::{BTreeMap, BTreeSet};

use semver::{Version, VersionReq};
use thiserror::Error;

/// How one package (or the root project) asks for another: either a semver
/// range resolved against the registry, or a git source pinned to a tag/rev.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepRequirement {
    Range(VersionReq),
    Git { url: String, git_ref: String },
}

impl DepRequirement {
    /// How this requirement should read in a conflict message.
    fn describe(&self) -> String {
        match self {
            DepRequirement::Range(req) => req.to_string(),
            DepRequirement::Git { url, git_ref } => format!("git+{url}@{git_ref}"),
        }
    }
}

/// The name used for the top-level project in conflict messages, so a user
/// can tell "your own laplace.toml wants this" from "some library wants it".
pub const ROOT_REQUIRER: &str = "this project";

/// Everything the resolver needs to know about the world outside itself.
/// The real implementation reads the filesystem registry and git; the tests
/// use an in-memory table.
pub trait PackageProvider {
    /// Every version of `name` that could be installed, in any order. An
    /// unknown package yields an empty list, not an error.
    fn available_versions(&self, name: &str) -> Result<Vec<Version>, GraphError>;

    /// Resolve a git requirement to the concrete version its manifest
    /// declares. May fetch.
    fn git_version(&self, name: &str, url: &str, git_ref: &str) -> Result<Version, GraphError>;

    /// The dependencies `name@version` declares in its own manifest.
    fn dependencies_of(
        &self,
        name: &str,
        version: &Version,
    ) -> Result<Vec<(String, DepRequirement)>, GraphError>;
}

/// One node of a successfully resolved graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPackage {
    pub name: String,
    pub version: Version,
    /// Whether the top-level project depends on this package directly (and
    /// may therefore call `pkg::func` on it -- see the encapsulation rule
    /// in `codegen`).
    pub direct: bool,
    /// Names of the packages this one depends on, sorted. Every entry is
    /// itself a node of the same resolved graph.
    pub dependencies: Vec<String>,
}

/// A resolved dependency graph: one version per package name, plus which of
/// them the root project asked for directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedGraph {
    /// Every reachable package, keyed by name (so iteration is sorted and
    /// therefore deterministic).
    pub packages: BTreeMap<String, ResolvedPackage>,
    /// The root project's direct dependency names, sorted.
    pub roots: Vec<String>,
}

impl ResolvedGraph {
    /// Every package reachable from `roots`, in dependency-first
    /// (topological) order: a package always appears after everything it
    /// depends on. Ties are broken alphabetically, so the order is stable
    /// across runs -- required for byte-identical build output.
    ///
    /// Only meaningful on a graph that passed [`detect_cycle`]; a cyclic
    /// graph would have been rejected before reaching here, but this falls
    /// back to appending any leftover nodes alphabetically rather than
    /// looping forever.
    pub fn topological_order(&self) -> Vec<&ResolvedPackage> {
        let mut emitted: BTreeSet<&str> = BTreeSet::new();
        let mut out: Vec<&ResolvedPackage> = Vec::new();

        // Repeatedly emit every not-yet-emitted node whose dependencies are
        // all emitted, alphabetically within each round.
        loop {
            let ready: Vec<&ResolvedPackage> = self
                .packages
                .values()
                .filter(|p| !emitted.contains(p.name.as_str()))
                .filter(|p| {
                    p.dependencies
                        .iter()
                        .all(|d| emitted.contains(d.as_str()) || !self.packages.contains_key(d))
                })
                .collect();
            if ready.is_empty() {
                break;
            }
            for pkg in ready {
                emitted.insert(pkg.name.as_str());
                out.push(pkg);
            }
        }

        for pkg in self.packages.values() {
            if !emitted.contains(pkg.name.as_str()) {
                out.push(pkg);
            }
        }
        out
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GraphError {
    #[error("package `{package}` was not found: no versions are available")]
    PackageUnavailable { package: String },

    #[error(
        "cannot pick one version of `{package}`: {}{}",
        format_requirers(.requirers),
        format_available(.available),
    )]
    VersionConflict {
        package: String,
        /// `(requirer, requirement)` pairs, sorted, so the message is
        /// deterministic.
        requirers: Vec<(String, String)>,
        available: Vec<String>,
    },

    #[error(
        "`{package}` is required both from git ({first}) and from git ({second}) -- a build \
         can only contain one copy of a package"
    )]
    GitSourceConflict {
        package: String,
        first: String,
        second: String,
    },

    #[error("dependency cycle: {}", .path.join(" -> "))]
    Cycle { path: Vec<String> },

    #[error(
        "dependency resolution did not settle after {rounds} rounds -- this is a bug in \
         laplace's resolver; please report the dependency set that triggered it"
    )]
    DidNotConverge { rounds: usize },

    /// Anything the provider itself failed at (registry I/O, a git fetch, a
    /// malformed package manifest). Kept as a plain string so this module
    /// stays independent of `ResolveError`.
    #[error("{0}")]
    Provider(String),
}

fn format_requirers(requirers: &[(String, String)]) -> String {
    requirers
        .iter()
        .map(|(who, req)| format!("\n  {who} requires `{req}`"))
        .collect::<Vec<_>>()
        .join("")
}

fn format_available(available: &[String]) -> String {
    if available.is_empty() {
        "\n  (no versions of it are available at all)".to_string()
    } else {
        format!("\n  available versions: {}", available.join(", "))
    }
}

/// One requirement on a package, and who asked for it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Constraint {
    requirer: String,
    requirement: DepRequirement,
}

/// Safety valve for the fixpoint loop below. Real graphs settle in
/// `depth + 1` rounds; anything near this bound means the resolver is
/// oscillating, which is a bug rather than a user error.
const MAX_ROUNDS: usize = 100;

/// Resolve `roots` (the top-level project's `[dependencies]`) plus every
/// transitive dependency into exactly one version per package name.
///
/// The algorithm is a fixpoint iteration: each round rebuilds the *entire*
/// constraint set from the currently-selected versions and re-picks the best
/// version for every constrained name. Rebuilding from scratch each round is
/// what keeps a stale constraint -- one contributed by a version that has
/// since been upgraded away -- from causing a phantom conflict.
pub fn resolve_graph(
    roots: &[(String, DepRequirement)],
    provider: &dyn PackageProvider,
) -> Result<ResolvedGraph, GraphError> {
    resolve_graph_with_preferences(roots, provider, &BTreeMap::new())
}

/// [`resolve_graph`], but sticky: whenever `preferred` names a version that
/// is still available and still satisfies every constraint, that version
/// wins over the highest matching one.
///
/// This is what makes `laplace add`/`update` behave: they pass the versions
/// already pinned in `laplace.lock` as preferences (minus the one package
/// being moved), so adding one dependency does not silently bump every
/// other dependency to its newest release.
pub fn resolve_graph_with_preferences(
    roots: &[(String, DepRequirement)],
    provider: &dyn PackageProvider,
    preferred: &BTreeMap<String, Version>,
) -> Result<ResolvedGraph, GraphError> {
    let root_names: BTreeSet<String> = roots.iter().map(|(name, _)| name.clone()).collect();
    let mut selected: BTreeMap<String, Version> = BTreeMap::new();

    for _round in 0..MAX_ROUNDS {
        let constraints = collect_constraints(roots, &selected, provider)?;

        let mut next: BTreeMap<String, Version> = BTreeMap::new();
        for (name, constraints) in &constraints {
            next.insert(
                name.clone(),
                pick_version(name, constraints, provider, preferred.get(name))?,
            );
        }

        if next == selected {
            let graph = build_graph(&selected, &root_names, provider)?;
            if let Some(path) = detect_cycle(&graph) {
                return Err(GraphError::Cycle { path });
            }
            return Ok(graph);
        }
        selected = next;
    }

    Err(GraphError::DidNotConverge { rounds: MAX_ROUNDS })
}

/// Rebuild the full constraint set: the root requirements, plus the
/// requirements declared by every currently-selected package.
fn collect_constraints(
    roots: &[(String, DepRequirement)],
    selected: &BTreeMap<String, Version>,
    provider: &dyn PackageProvider,
) -> Result<BTreeMap<String, Vec<Constraint>>, GraphError> {
    let mut constraints: BTreeMap<String, Vec<Constraint>> = BTreeMap::new();

    for (name, requirement) in roots {
        constraints
            .entry(name.clone())
            .or_default()
            .push(Constraint {
                requirer: ROOT_REQUIRER.to_string(),
                requirement: requirement.clone(),
            });
    }

    for (name, version) in selected {
        for (dep_name, requirement) in provider.dependencies_of(name, version)? {
            constraints
                .entry(dep_name)
                .or_default()
                .push(Constraint {
                    requirer: format!("`{name}@{version}`"),
                    requirement,
                });
        }
    }

    // Constraints from a package that is no longer reachable (its requirer
    // was dropped in an earlier round) are pruned implicitly: `selected` is
    // itself rebuilt from `constraints` every round, so an unreachable name
    // simply stops appearing.
    for list in constraints.values_mut() {
        list.sort_by(|a, b| {
            a.requirer
                .cmp(&b.requirer)
                .then_with(|| a.requirement.describe().cmp(&b.requirement.describe()))
        });
        list.dedup();
    }
    Ok(constraints)
}

/// Pick the single version of `name` satisfying every constraint on it:
/// the highest available version matching all ranges. Git requirements pin
/// the version outright (whatever the fetched manifest declares) and must
/// still satisfy any range constraints alongside them.
fn pick_version(
    name: &str,
    constraints: &[Constraint],
    provider: &dyn PackageProvider,
    preferred: Option<&Version>,
) -> Result<Version, GraphError> {
    let mut git: Option<(&str, &str, &str)> = None; // (url, ref, requirer)
    let mut ranges: Vec<(&VersionReq, &str)> = Vec::new();

    for constraint in constraints {
        match &constraint.requirement {
            DepRequirement::Range(req) => ranges.push((req, constraint.requirer.as_str())),
            DepRequirement::Git { url, git_ref } => match git {
                Some((prev_url, prev_ref, _)) if prev_url != url || prev_ref != git_ref => {
                    return Err(GraphError::GitSourceConflict {
                        package: name.to_string(),
                        first: format!("{prev_url}@{prev_ref}"),
                        second: format!("{url}@{git_ref}"),
                    })
                }
                _ => git = Some((url, git_ref, constraint.requirer.as_str())),
            },
        }
    }

    let candidates: Vec<Version> = match git {
        // A git source is its own registry-of-one: the checked-out ref
        // decides the version, and every range constraint has to accept it.
        Some((url, git_ref, _)) => vec![provider.git_version(name, url, git_ref)?],
        None => provider.available_versions(name)?,
    };

    if candidates.is_empty() {
        return Err(GraphError::PackageUnavailable {
            package: name.to_string(),
        });
    }

    let satisfies =
        |version: &Version| ranges.iter().all(|(req, _)| req.matches(version));

    // A still-valid preference wins over the newest match, so a lock stays
    // put unless something actually forces it to move.
    let best = preferred
        .filter(|version| candidates.contains(version) && satisfies(version))
        .or_else(|| candidates.iter().filter(|v| satisfies(v)).max());

    match best {
        Some(version) => Ok(version.clone()),
        None => {
            let mut requirers: Vec<(String, String)> = constraints
                .iter()
                .map(|c| (c.requirer.clone(), c.requirement.describe()))
                .collect();
            requirers.sort();
            requirers.dedup();
            let mut available: Vec<String> = candidates.iter().map(|v| v.to_string()).collect();
            available.sort();
            Err(GraphError::VersionConflict {
                package: name.to_string(),
                requirers,
                available,
            })
        }
    }
}

fn build_graph(
    selected: &BTreeMap<String, Version>,
    root_names: &BTreeSet<String>,
    provider: &dyn PackageProvider,
) -> Result<ResolvedGraph, GraphError> {
    let mut packages = BTreeMap::new();
    for (name, version) in selected {
        let mut dependencies: Vec<String> = provider
            .dependencies_of(name, version)?
            .into_iter()
            .map(|(dep_name, _)| dep_name)
            .collect();
        dependencies.sort();
        dependencies.dedup();

        packages.insert(
            name.clone(),
            ResolvedPackage {
                name: name.clone(),
                version: version.clone(),
                direct: root_names.contains(name),
                dependencies,
            },
        );
    }

    let mut roots: Vec<String> = root_names.iter().cloned().collect();
    roots.retain(|name| packages.contains_key(name));
    roots.sort();

    Ok(ResolvedGraph { packages, roots })
}

/// Find one cycle in `graph`, if any, and return it as a readable path
/// (`a -> b -> a`). Iterative depth-first search with an explicit stack, so
/// a pathological graph can't blow the real one.
pub fn detect_cycle(graph: &ResolvedGraph) -> Option<Vec<String>> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        InProgress,
        Done,
    }

    let mut marks: BTreeMap<&str, Mark> = BTreeMap::new();
    // Names are visited in sorted order so the *same* cycle is reported for
    // the same graph on every run.
    for start in graph.packages.keys() {
        if marks.get(start.as_str()) == Some(&Mark::Done) {
            continue;
        }

        // (node, index of the next dependency to walk into)
        let mut stack: Vec<(&str, usize)> = vec![(start.as_str(), 0)];
        marks.insert(start.as_str(), Mark::InProgress);

        while let Some(&(node, next)) = stack.last() {
            let deps: &[String] = graph
                .packages
                .get(node)
                .map(|pkg| pkg.dependencies.as_slice())
                .unwrap_or(&[]);

            if next >= deps.len() {
                marks.insert(node, Mark::Done);
                stack.pop();
                continue;
            }
            let dep = deps[next].as_str();
            stack.last_mut().expect("stack is non-empty").1 += 1;

            if !graph.packages.contains_key(dep) {
                continue;
            }
            match marks.get(dep) {
                Some(Mark::Done) => continue,
                Some(Mark::InProgress) => {
                    // `dep` is somewhere on the current stack; the cycle is
                    // that suffix of the stack plus `dep` closing the loop.
                    let at = stack.iter().position(|(n, _)| *n == dep).unwrap_or(0);
                    let mut path: Vec<String> =
                        stack[at..].iter().map(|(n, _)| n.to_string()).collect();
                    path.push(dep.to_string());
                    return Some(path);
                }
                None => {
                    marks.insert(dep, Mark::InProgress);
                    stack.push((dep, 0));
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory provider: `(name, version) -> dependencies`, no I/O.
    struct TestProvider {
        packages: BTreeMap<(String, Version), Vec<(String, DepRequirement)>>,
    }

    impl TestProvider {
        fn new() -> Self {
            TestProvider {
                packages: BTreeMap::new(),
            }
        }

        /// `add("regression", "1.0.0", &[("stats", "^1.0")])`
        fn add(mut self, name: &str, version: &str, deps: &[(&str, &str)]) -> Self {
            let deps = deps
                .iter()
                .map(|(dep, range)| {
                    (
                        dep.to_string(),
                        DepRequirement::Range(VersionReq::parse(range).unwrap()),
                    )
                })
                .collect();
            self.packages
                .insert((name.to_string(), Version::parse(version).unwrap()), deps);
            self
        }
    }

    impl PackageProvider for TestProvider {
        fn available_versions(&self, name: &str) -> Result<Vec<Version>, GraphError> {
            Ok(self
                .packages
                .keys()
                .filter(|(n, _)| n == name)
                .map(|(_, v)| v.clone())
                .collect())
        }

        fn git_version(&self, name: &str, _url: &str, git_ref: &str) -> Result<Version, GraphError> {
            let _ = name;
            Version::parse(git_ref).map_err(|e| GraphError::Provider(e.to_string()))
        }

        fn dependencies_of(
            &self,
            name: &str,
            version: &Version,
        ) -> Result<Vec<(String, DepRequirement)>, GraphError> {
            Ok(self
                .packages
                .get(&(name.to_string(), version.clone()))
                .cloned()
                .unwrap_or_default())
        }
    }

    fn root(name: &str, range: &str) -> (String, DepRequirement) {
        (
            name.to_string(),
            DepRequirement::Range(VersionReq::parse(range).unwrap()),
        )
    }

    fn version_of(graph: &ResolvedGraph, name: &str) -> String {
        graph
            .packages
            .get(name)
            .map(|p| p.version.to_string())
            .unwrap_or_else(|| "<absent>".to_string())
    }

    #[test]
    fn resolves_a_single_direct_dependency_to_the_latest_matching_version() {
        let provider = TestProvider::new()
            .add("stats", "1.0.0", &[])
            .add("stats", "1.4.0", &[])
            .add("stats", "2.0.0", &[]);

        let graph = resolve_graph(&[root("stats", "^1.0")], &provider).unwrap();
        assert_eq!(version_of(&graph, "stats"), "1.4.0");
        assert_eq!(graph.roots, vec!["stats"]);
        assert!(graph.packages["stats"].direct);
    }

    #[test]
    fn pulls_in_a_transitive_dependency_the_project_never_named() {
        let provider = TestProvider::new()
            .add("regression", "1.0.0", &[("stats", "^1.0")])
            .add("stats", "1.2.0", &[]);

        let graph = resolve_graph(&[root("regression", "^1.0")], &provider).unwrap();

        assert_eq!(graph.roots, vec!["regression"]);
        assert_eq!(version_of(&graph, "stats"), "1.2.0");
        // `stats` is in the build but is *not* a direct dependency -- the
        // project may not call `stats::f()` itself (encapsulation).
        assert!(!graph.packages["stats"].direct);
        assert_eq!(graph.packages["regression"].dependencies, vec!["stats"]);
    }

    #[test]
    fn resolves_a_deep_chain() {
        let provider = TestProvider::new()
            .add("a", "1.0.0", &[("b", "^1.0")])
            .add("b", "1.0.0", &[("c", "^1.0")])
            .add("c", "1.0.0", &[("d", "^1.0")])
            .add("d", "1.0.0", &[]);

        let graph = resolve_graph(&[root("a", "^1.0")], &provider).unwrap();
        assert_eq!(graph.packages.len(), 4);

        let order: Vec<&str> = graph
            .topological_order()
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(order, vec!["d", "c", "b", "a"]);
    }

    #[test]
    fn diamond_with_overlapping_ranges_unifies_to_one_version() {
        let provider = TestProvider::new()
            .add("regression", "1.0.0", &[("stats", ">=1.1, <2.0")])
            .add("stats", "1.0.0", &[])
            .add("stats", "1.5.0", &[])
            .add("stats", "2.0.0", &[]);

        let graph = resolve_graph(
            &[root("regression", "^1.0"), root("stats", "^1.0")],
            &provider,
        )
        .unwrap();

        // One `stats`, satisfying both `^1.0` and `>=1.1, <2.0`.
        assert_eq!(graph.packages.len(), 2);
        assert_eq!(version_of(&graph, "stats"), "1.5.0");
        assert!(graph.packages["stats"].direct);
    }

    #[test]
    fn diamond_with_incompatible_ranges_names_both_requirers() {
        let provider = TestProvider::new()
            .add("regression", "1.0.0", &[("stats", "^2.0")])
            .add("stats", "1.0.0", &[])
            .add("stats", "2.0.0", &[]);

        let err = resolve_graph(
            &[root("regression", "^1.0"), root("stats", "^1.0")],
            &provider,
        )
        .unwrap_err();

        let GraphError::VersionConflict {
            package, requirers, ..
        } = &err
        else {
            panic!("expected a version conflict, got {err:?}");
        };
        assert_eq!(package, "stats");

        let rendered = err.to_string();
        assert!(rendered.contains("this project"), "{rendered}");
        assert!(rendered.contains("regression@1.0.0"), "{rendered}");
        assert!(rendered.contains("^1"), "{rendered}");
        assert!(rendered.contains("^2"), "{rendered}");
        assert_eq!(requirers.len(), 2);
    }

    #[test]
    fn a_package_with_no_available_versions_is_reported_as_unavailable() {
        let provider = TestProvider::new();
        let err = resolve_graph(&[root("ghost", "^1.0")], &provider).unwrap_err();
        assert_eq!(
            err,
            GraphError::PackageUnavailable {
                package: "ghost".to_string()
            }
        );
    }

    #[test]
    fn upgrading_a_package_drops_the_constraints_of_the_version_it_replaced() {
        // `app` starts out resolved at 1.0.0 (which pins `stats` to ^1),
        // but 2.0.0 is available and wanted, and it needs `stats` ^2. The
        // stale ^1 constraint must not survive into the final round.
        let provider = TestProvider::new()
            .add("app", "1.0.0", &[("stats", "^1.0")])
            .add("app", "2.0.0", &[("stats", "^2.0")])
            .add("stats", "1.0.0", &[])
            .add("stats", "2.0.0", &[]);

        let graph = resolve_graph(&[root("app", ">=1.0")], &provider).unwrap();
        assert_eq!(version_of(&graph, "app"), "2.0.0");
        assert_eq!(version_of(&graph, "stats"), "2.0.0");
    }

    #[test]
    fn detects_a_direct_self_cycle() {
        let provider = TestProvider::new().add("a", "1.0.0", &[("a", "^1.0")]);
        let err = resolve_graph(&[root("a", "^1.0")], &provider).unwrap_err();
        assert_eq!(
            err,
            GraphError::Cycle {
                path: vec!["a".to_string(), "a".to_string()]
            }
        );
    }

    #[test]
    fn detects_an_indirect_two_package_cycle() {
        let provider = TestProvider::new()
            .add("a", "1.0.0", &[("b", "^1.0")])
            .add("b", "1.0.0", &[("a", "^1.0")]);

        let err = resolve_graph(&[root("a", "^1.0")], &provider).unwrap_err();
        let GraphError::Cycle { path } = &err else {
            panic!("expected a cycle, got {err:?}");
        };
        assert_eq!(path, &vec!["a".to_string(), "b".to_string(), "a".to_string()]);
        assert_eq!(err.to_string(), "dependency cycle: a -> b -> a");
    }

    #[test]
    fn detects_a_deeper_cycle_that_hangs_off_an_acyclic_prefix() {
        let provider = TestProvider::new()
            .add("app", "1.0.0", &[("a", "^1.0")])
            .add("a", "1.0.0", &[("b", "^1.0")])
            .add("b", "1.0.0", &[("c", "^1.0")])
            .add("c", "1.0.0", &[("a", "^1.0")]);

        let err = resolve_graph(&[root("app", "^1.0")], &provider).unwrap_err();
        let GraphError::Cycle { path } = err else {
            panic!("expected a cycle");
        };
        assert_eq!(
            path,
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "a".to_string()
            ]
        );
    }

    #[test]
    fn an_acyclic_diamond_is_not_reported_as_a_cycle() {
        let provider = TestProvider::new()
            .add("app", "1.0.0", &[("left", "^1.0"), ("right", "^1.0")])
            .add("left", "1.0.0", &[("base", "^1.0")])
            .add("right", "1.0.0", &[("base", "^1.0")])
            .add("base", "1.0.0", &[]);

        let graph = resolve_graph(&[root("app", "^1.0")], &provider).unwrap();
        assert!(detect_cycle(&graph).is_none());

        let order: Vec<&str> = graph
            .topological_order()
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(order, vec!["base", "left", "right", "app"]);
    }

    #[test]
    fn topological_order_is_stable_across_runs() {
        let provider = TestProvider::new()
            .add("app", "1.0.0", &[("zeta", "^1.0"), ("alpha", "^1.0")])
            .add("zeta", "1.0.0", &[])
            .add("alpha", "1.0.0", &[]);

        let first = resolve_graph(&[root("app", "^1.0")], &provider).unwrap();
        let second = resolve_graph(&[root("app", "^1.0")], &provider).unwrap();
        let names = |g: &ResolvedGraph| {
            g.topological_order()
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&first), names(&second));
        assert_eq!(names(&first), vec!["alpha", "zeta", "app"]);
    }

    #[test]
    fn a_git_requirement_pins_the_version_and_must_satisfy_range_constraints() {
        let provider = TestProvider::new()
            .add("app", "1.0.0", &[("stats", "^1.0")])
            .add("stats", "1.3.0", &[]);

        // The root asks for `stats` from git at ref "1.3.0" (the test
        // provider reads the version straight off the ref), and `app` also
        // wants `stats` ^1.0 -- compatible, so it resolves.
        let roots = vec![
            root("app", "^1.0"),
            (
                "stats".to_string(),
                DepRequirement::Git {
                    url: "https://example.com/stats".to_string(),
                    git_ref: "1.3.0".to_string(),
                },
            ),
        ];
        let graph = resolve_graph(&roots, &provider).unwrap();
        assert_eq!(version_of(&graph, "stats"), "1.3.0");

        // ...and an incompatible range against the same git pin conflicts.
        let provider = TestProvider::new()
            .add("app", "1.0.0", &[("stats", "^2.0")])
            .add("stats", "1.3.0", &[]);
        let err = resolve_graph(&roots, &provider).unwrap_err();
        assert!(matches!(err, GraphError::VersionConflict { .. }), "{err:?}");
    }

    #[test]
    fn two_different_git_refs_for_one_package_conflict() {
        let provider = TestProvider::new();
        let roots = vec![
            (
                "stats".to_string(),
                DepRequirement::Git {
                    url: "https://example.com/stats".to_string(),
                    git_ref: "1.0.0".to_string(),
                },
            ),
            (
                "stats".to_string(),
                DepRequirement::Git {
                    url: "https://example.com/stats".to_string(),
                    git_ref: "2.0.0".to_string(),
                },
            ),
        ];
        let err = resolve_graph(&roots, &provider).unwrap_err();
        assert!(matches!(err, GraphError::GitSourceConflict { .. }), "{err:?}");
    }

    #[test]
    fn resolution_is_deterministic_for_the_same_inputs() {
        let provider = TestProvider::new()
            .add("regression", "1.0.0", &[("stats", "^1.0")])
            .add("regression", "1.1.0", &[("stats", "^1.0")])
            .add("stats", "1.0.0", &[])
            .add("stats", "1.9.0", &[]);

        let a = resolve_graph(&[root("regression", "^1.0")], &provider).unwrap();
        let b = resolve_graph(&[root("regression", "^1.0")], &provider).unwrap();
        assert_eq!(a, b);
    }
}
