//! `laplace-sample`. Usually reached through the compiler as
//! `laplace sample ...` / `laplace sampler ...`, which forward their
//! arguments here unchanged (D1).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

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
    /// Work with .lapsampler files
    Sampler {
        #[command(subcommand)]
        action: SamplerAction,
    },
}

#[derive(Subcommand)]
enum SamplerAction {
    /// Check a .lapsampler: syntax, every kernel resolves (built-ins,
    /// local .lapkernel files, which are checked too), settings, the
    /// adapt() contracts. With --model, also that every model parameter is
    /// in exactly one block (coverage, D2).
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
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Command::Sampler {
            action: SamplerAction::Check { file, model, data },
        } => match check_sampler(&file, model.as_deref(), data.as_deref()) {
            Ok(report) => {
                println!("{report}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e:#}");
                ExitCode::FAILURE
            }
        },
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
