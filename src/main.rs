mod lattice;
mod sim;
mod visual;
mod setup;
mod store;

use crate::lattice::ScalarLattice4D;
use crate::sim::{InitialState, System, SystemBuilder};
use plotters::prelude::*;
use crate::visual::{plot_observable, GraphData, GraphDesc};

fn main() -> anyhow::Result<()> {
    let mut sim = SystemBuilder::new()
        .sizes([40, 25, 25, 25])
        .spacing(1.0)
        .initial_step_size(0.1)
        .upper_acceptance(0.5)
        .lower_acceptance(0.3)
        .mass_squared(1.0)
        .coupling(0.5)
        .initial_value(InitialState::RandomRange(-0.5..0.5))
        .th_block_size(100)
        .th_threshold(0.005)
        .build()?;

    visual::plot_lattice(0, sim.lattice())?;

    let total_sweeps = 50000;
    sim.simulate_checkerboard(total_sweeps);

    println!("Printing lattice");
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
            GraphData {
                caption: "2-point correlator",
                xdata: &tdata,
                ydata: sim.correlator2(),
                ..Default::default()
            },
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
