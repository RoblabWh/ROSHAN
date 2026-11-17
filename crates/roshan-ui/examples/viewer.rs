//! The viewer without Python (so no saving, no terrain fetch, no learned policy):
//!     cargo run --release -p roshan-ui --example viewer            # demo scenario, greedy planner
//!     cargo run --release -p roshan-ui --example viewer navigate   # the fly task, steer by hand
//!     cargo run --release -p roshan-ui --example viewer firespin   # demo with the particle fire model

use roshan_core::navigate::NavConfig;
use roshan_core::terrain::Grid;
use roshan_ui::{Hooks, Options, Session};

fn main() {
    let arg = std::env::args().nth(1);
    let opts = Options {
        fire_model: match arg.as_deref() {
            Some("firespin") => roshan_core::fire::FireModel::Firespin,
            _ => Default::default(),
        },
        ..Options::default()
    };
    let session = match arg.as_deref() {
        Some("navigate") => Session::Navigate {
            grid: Grid {
                nx: 35,
                ny: 36,
                cell_m: 10.0,
            },
            cfg: NavConfig::default(),
        },
        _ => Session::Scenario(roshan_ui::demo_scenario()),
    };
    println!(
        "{:?}",
        roshan_ui::run(session, Hooks::default(), opts, &mut || true)
    );
}
