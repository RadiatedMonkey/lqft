mod lattice;
mod sim;
mod visual;
mod setup;
mod snapshot;
mod stats;

use std::time::UNIX_EPOCH;
use plotters::prelude::*;
use crate::setup::{AcceptanceDesc, BurnInDesc, InitialState, LatticeDesc, SnapshotDesc, SnapshotType, SystemBuilder};
use crate::snapshot::FlushMethod;
use crate::visual::{plot_observable, GraphData, GraphDesc};

fn main() -> anyhow::Result<()> {
    let mut sim = SystemBuilder::new()
        .mass_squared(1.0)
        .coupling(0.5)
        .enable_snapshot(SnapshotDesc {
            file: "snapshots/snapshots.h5".to_string(),
            ty: SnapshotType::Checkpoint,
            chunk_size: [16; 4],
            flush_method: FlushMethod::Sequential,
        })
        .with_lattice(LatticeDesc {
            dimensions: [40, 20, 20, 20],
            initial_state: InitialState::RandomRange(-0.5..0.5),
            spacing: 1.0
        })
        .with_acceptance(AcceptanceDesc {
            correction_interval: 20_000,
            initial_step_size: 1.0,
            desired_range: 0.3..0.5,
            correction_size: 0.05
        })
        .with_burn_in(BurnInDesc {
            avg_block_size: 10,
            desired_ratio: 0.05
        })
        .build()?;

    visual::plot_lattice(0, sim.lattice())?;

    let total_sweeps = 50;
    sim.simulate_checkerboard(total_sweeps)?;

    visual::plot_lattice(1, sim.lattice())?;

    let stats = sim.stats();

    let sweepx = (0..total_sweeps).map(|i| i as f64).collect::<Vec<_>>();

    let variance = stats
        .mean_history
        .iter()
        .zip(stats.meansq_history.iter())
        .map(|(&mean, &meansq)| meansq - mean.powf(2.0))
        .collect::<Vec<_>>();

    let tdata = (0..sim.lattice().dimensions()[0]).map(|v| v as f64).collect::<Vec<_>>();
    // let corr2 = sim.correlator2();

    let desc = GraphDesc {
        dimensions: (2000, 2000),
        file: "layout.png",
        layout: (3, 3),
        burnin_time: 0,
        graphs: &[
            GraphData {
                caption: "Mean value",
                xdata: &sweepx,
                ydata: &stats.mean_history,
                ..Default::default()
            },
            GraphData {
                caption: "Variance",
                xdata: &sweepx,
                ydata: &variance,
                ..Default::default()
            },
            GraphData {
                caption: "Mean squared",
                xdata: &sweepx,
                ydata: &stats.meansq_history,
                ..Default::default()
            },
            GraphData {
                caption: "Action block average",
                xdata: &sweepx[sim.th_block_size() * 2..],
                ydata: &stats.thermalisation_ratio_history,
                ..Default::default()
            },
            GraphData {
                caption: "Action",
                xdata: &sweepx,
                ydata: &stats.action_history,
                ..Default::default()
            },
            GraphData {
                caption: "Step size",
                xdata: &sweepx,
                ydata: &stats.step_size_history,
                ylim: 0.75..1.25,
                ..Default::default()
            },
            GraphData {
                caption: "Acceptance ratio",
                xdata: &sweepx,
                ydata: &stats.accept_ratio_history,
                ylim: 20.0..80.0,
                ..Default::default()
            },
            // GraphData {
            //     caption: "2-point correlator",
            //     xdata: &tdata,
            //     ydata: sim.correlator2(),
            //     ..Default::default()
            // }, // FIXME: Correlator2 seems to be filled with NaNs
            // GraphData {
            //     caption: "2-point Function",
            //     xdata: &tdata,
            //     ydata: &corr2,
            //     ..Default::default()
            // }
        ],
    };
    plot_observable(desc, &sim)?;

    println!("System thermalised at sweep {:?}", sim.stats().thermalised_at);

    Ok(())
}
