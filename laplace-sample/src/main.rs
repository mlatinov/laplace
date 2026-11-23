//! `laplace-sample`. Usually reached through the compiler as
//! `laplace sample ...`, which forwards its arguments here unchanged (D1).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use laplace_sample::runner::BuiltIn;
use laplace_sample::sample::{display, sample, SampleArgs};

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
    /// Sample a .laplace or .stan model with a built-in sampler and write
    /// one CmdStan-style CSV per chain (readable with
    /// cmdstanr::as_cmdstan_fit()). Prints the CSV paths, one per line.
    Sample {
        /// The model: a .laplace file (compiled with `laplace build` first)
        /// or a .stan file
        model: PathBuf,
        /// Data as a JSON file (cmdstanr::write_stan_json format)
        #[arg(long)]
        data: Option<PathBuf>,
        /// Built-in sampler: nuts, rwm, mala or hmc
        #[arg(long, default_value = "nuts")]
        sampler: String,
        #[arg(long, default_value_t = 4)]
        chains: u64,
        /// Chains run at the same time (default: all of them)
        #[arg(long)]
        parallel_chains: Option<u64>,
        /// Warmup (adaptation) iterations per chain; not saved
        #[arg(long, default_value_t = 1000)]
        warmup: u64,
        /// Saved draws per chain
        #[arg(long, default_value_t = 1000)]
        draws: u64,
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
        /// adapt. hmc: the same plus steps, jitter.
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
        /// Directory for the CSV files (`<model>-<chain>.csv`)
        #[arg(long, default_value = ".")]
        output_dir: PathBuf,
    },
}

fn main() -> ExitCode {
    let Command::Sample {
        model,
        data,
        sampler,
        chains,
        parallel_chains,
        warmup,
        draws,
        seed,
        init,
        set,
        output_dir,
    } = Cli::parse().command;
    let args = SampleArgs {
        model,
        data,
        sampler,
        chains,
        parallel_chains,
        warmup,
        draws,
        seed,
        init,
        set,
        output_dir,
    };
    // Fail on a bad --sampler before compiling anything.
    if let Err(e) = BuiltIn::parse(&args.sampler) {
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
