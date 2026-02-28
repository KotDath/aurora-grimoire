mod cli;

use clap::Parser;

fn main() {
    let app = cli::App::parse();
    cli::run(app);
}
