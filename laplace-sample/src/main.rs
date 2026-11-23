//! `laplace-sample`. Usually reached through the compiler as
//! `laplace sample ...` / `laplace sampler ...`, which forward their
//! arguments here unchanged (D1).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use laplace_sample::authoring;
use laplace_sample::runner::Sampler;
use laplace_sample::sample::{check_sampler, display, sample, SampleArgs};

#[derive(Parser)]
#[command(
    name = "laplace-sample",
    version,
    about = "Custom samplers for laplace models (BridgeStan backend)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Sample a .laplace or .stan model and write one CmdStan-style CSV per
    /// chain (readable with cmdstanr::as_cmdstan_fit()). Prints the CSV
    /// paths, one per line.
    Sample {
        /// The model: a .laplace file (compiled with `laplace build` first)
        /// or a .stan file
        model: PathBuf,
        /// Data as a JSON file (cmdstanr::write_stan_json format)
        #[arg(long)]
        data: Option<PathBuf>,
        /// A built-in sampler (nuts, rwm, mala, hmc), a .lapkernel file, or
        /// a .lapsampler file composing kernels over parameter blocks
        #[arg(long, default_value = "nuts")]
        sampler: String,
        #[arg(long, default_value_t = 4)]
        chains: u64,
        /// Chains run at the same time (default: all of them)
        #[arg(long)]
        parallel_chains: Option<u64>,
        /// Warmup iterations per chain, not saved (default: the
        /// .lapsampler's, else 1000)
        #[arg(long)]
        warmup: Option<u64>,
        /// Iterations per chain after warmup (default: the .lapsampler's,
        /// else 1000)
        #[arg(long)]
        draws: Option<u64>,
        /// Save every N-th draw (default: the .lapsampler's, else 1)
        #[arg(long)]
        thin: Option<u64>,
        /// Random seed (default: random, recorded in the CSV header)
        #[arg(long)]
        seed: Option<u64>,
        /// Initial values: a radius R for Uniform(-R, R) on the
        /// unconstrained scale (default 2), or a JSON file of parameter
        /// values; repeat once per chain for per-chain files
        #[arg(long)]
        init: Vec<String>,
        /// Sampler setting as key=value, repeatable. nuts: step_size,
        /// target_accept, max_depth. rwm, mala: step_size, target_accept,
        /// adapt. hmc: the same plus steps, jitter. A .lapkernel file: its
        /// int and real tuning values, adapt, and target_accept if it has
        /// an @adapt(step_size) value (default: the tag's). Not for a
        /// .lapsampler, whose settings are in the file.
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
        /// Directory for the CSV files (`<model>-<chain>.csv`)
        #[arg(long, default_value = ".")]
        output_dir: PathBuf,
    },
    /// Work with .lapsampler files and kernels
    Sampler {
        #[command(subcommand)]
        action: SamplerAction,
    },
    /// Scaffold something new
    New {
        #[command(subcommand)]
        what: NewWhat,
    },
}

#[derive(Subcommand)]
enum NewWhat {
    /// A kernel. Outside a kernel package: a new package `<name>/` with
    /// laplace.toml, kernels/<name>.lapkernel, tests/<name>.test.toml and
    /// README.md. Inside one: only the kernel and its test file.
    Kernel {
        name: String,
        /// `proposal` (ends in propose(), Metropolis-Hastings) or `direct`
        /// (ends in move())
        #[arg(long, default_value = "proposal", value_parser = ["proposal", "direct"])]
        kind: String,
        /// The new package's name, when creating one (default: the
        /// kernel's name). Inside a kernel package it must be that package.
        #[arg(long)]
        package: Option<String>,
    },
}

#[derive(Subcommand)]
enum SamplerAction {
    /// Check a .lapsampler: syntax, every kernel resolves (built-ins,
    /// local .lapkernel files and library kernels, which are checked too),
    /// settings, the adapt() contracts. With --model, also that every model
    /// parameter is in exactly one block (coverage, D2). Given a .lapkernel,
    /// check that one kernel.
    Check {
        file: PathBuf,
        /// The model whose parameters the blocks must cover (.laplace or
        /// .stan); compiled with BridgeStan, cached
        #[arg(long)]
        model: Option<PathBuf>,
        /// The model's data, if it has any (sizes come from it)
        #[arg(long)]
        data: Option<PathBuf>,
    },
    /// A kernel's documentation: doc comments, tuning values with their
    /// @adapt tags, and whether its step ends in propose() or move().
    /// `<package>::<kernel>` (an installed kernel package; `laplace doc`
    /// forwards here) or a .lapkernel file.
    Doc { spec: String },
    /// The built-in kernels and the kernels of this project's kernel
    /// packages, with their settings and tuning values
    List,
    /// Run a kernel or sampler on the reference suite: standard normal,
    /// correlated Gaussian and non-centred eight schools must pass; Neal's
    /// funnel, a banana and a bimodal mixture are reported. Prints a table
    /// with ESS per gradient and per second and the ratio to plain NUTS;
    /// exits non-zero when a must-pass target does not pass.
    Test {
        /// A built-in (nuts, rwm, mala, hmc), `<package>::<kernel>`, a
        /// .lapkernel or a .lapsampler
        sampler: String,
        /// Draw counts and settings (default: the kernel's
        /// tests/<name>.test.toml, or <name>.test.toml beside a
        /// .lapsampler)
        #[arg(long)]
        config: Option<PathBuf>,
        /// A setting or tuning value as key=value, after the file's [set]
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
        /// Only one tier: `must-pass` or `report-only`
        #[arg(long, value_parser = ["must-pass", "report-only"])]
        tier: Option<String>,
        #[arg(long, default_value_t = 20261123)]
        seed: u64,
        /// Skip the plain-NUTS runs (no ratios)
        #[arg(long)]
        no_compare: bool,
    },
}

fn main() -> ExitCode {
    let report = |r: anyhow::Result<String>| match r {
        Ok(out) => {
            println!("{}", out.trim_end());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e:#}");
            ExitCode::FAILURE
        }
    };
    let project = laplace_sample::libraries::Project::current;
    match Cli::parse().command {
        Command::Sampler {
            action: SamplerAction::Check { file, model, data },
        } => {
            if file.extension().and_then(|e| e.to_str()) == Some("lapkernel") {
                report(authoring::check_kernel(&file))
            } else {
                report(check_sampler(&file, model.as_deref(), data.as_deref()))
            }
        }
        Command::Sampler {
            action: SamplerAction::Doc { spec },
        } => report(authoring::doc(&spec, &project())),
        Command::Sampler {
            action: SamplerAction::List,
        } => report(Ok(authoring::list(&project()))),
        Command::Sampler {
            action:
                SamplerAction::Test {
                    sampler,
                    config,
                    set,
                    tier,
                    seed,
                    no_compare,
                },
        } => sampler_test(
            &sampler,
            config.as_deref(),
            &set,
            tier.as_deref(),
            seed,
            !no_compare,
        ),
        Command::New {
            what:
                NewWhat::Kernel {
                    name,
                    kind,
                    package,
                },
        } => {
            let kind = if kind == "direct" {
                authoring::KernelKind::Direct
            } else {
                authoring::KernelKind::Proposal
            };
            let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            report(
                authoring::new_kernel(&name, kind, &cwd, package.as_deref()).map(|files| {
                    let shown: Vec<String> = files
                        .iter()
                        .map(|f| {
                            format!("  created {}", f.strip_prefix(&cwd).unwrap_or(f).display())
                        })
                        .collect();
                    format!(
                        "{}\ncheck it with `laplace sampler check {}`",
                        shown.join("\n"),
                        files
                            .iter()
                            .find(|f| f.extension().and_then(|e| e.to_str()) == Some("lapkernel"))
                            .map(|f| f.strip_prefix(&cwd).unwrap_or(f).display().to_string())
                            .unwrap_or_default()
                    )
                }),
            )
        }
        Command::Sample {
            model,
            data,
            sampler,
            chains,
            parallel_chains,
            warmup,
            draws,
            thin,
            seed,
            init,
            set,
            output_dir,
        } => {
            let args = SampleArgs {
                model,
                data,
                sampler,
                chains,
                parallel_chains,
                warmup,
                draws,
                thin,
                seed,
                init,
                set,
                output_dir,
            };
            // Fail on a bad --sampler (or a .lapkernel / .lapsampler that
            // does not parse or check) before compiling anything.
            if let Err(e) = Sampler::from_arg(&args.sampler, &args.set) {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
            match sample(&args) {
                Ok(paths) => {
                    for p in paths {
                        println!("{}", display(&p));
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e:#}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}

/// `laplace sampler test`: exit 0 when every must-pass target passes.
fn sampler_test(
    spec: &str,
    config: Option<&std::path::Path>,
    sets: &[String],
    tier: Option<&str>,
    seed: u64,
    compare: bool,
) -> ExitCode {
    use laplace_sample::suite;
    let project = laplace_sample::libraries::Project::current();
    let prepared = match suite::prepare(spec, sets, config, &project) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("settings: {}; seed {seed}", prepared.config_source);
    let req = suite::Request {
        sampler: &prepared.sampler,
        config: &prepared.config,
        tier: tier.map(|t| {
            if t == "must-pass" {
                suite::Tier::MustPass
            } else {
                suite::Tier::ReportOnly
            }
        }),
        seed,
        compare,
    };
    match suite::run_suite(&req, |t| eprintln!("running {t} ...")) {
        Ok(rows) => {
            print!("{}", suite::render(spec, &rows));
            if suite::passed(&rows) {
                ExitCode::SUCCESS
            } else {
                println!(
                    "\nresult: a must-pass target did not pass (or the sampler runs on none of them: test its \
                     kernels instead)"
                );
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
