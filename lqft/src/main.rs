#![feature(portable_simd)]

mod lattice;
mod observable;
mod setup;
mod sim;
mod snapshot;
mod stats;
mod metrics;
mod observable_impl;

use crate::setup::{AcceptanceDesc, BurnInDesc, FlushMethod, InitialState, LatticeCreateDesc, LatticeDesc, LatticeIterMethod, LatticeLoadDesc, ParamDesc, PerformanceDesc, SnapshotDesc, SnapshotLocation, SnapshotType, SystemBuilder};
use std::process::ExitCode;
use tracing_loki::url::Url;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use crate::observable_impl::{MeanValue, Variance};

/// Makes all struct fields public in the current and specified modules.
/// This makes it easier to spread implementation details over multiple archive.
#[macro_export]
macro_rules! all_public_in {
    ($module:path, $vis:vis struct $name:ident {
        $(
            $(#[$meta:meta])*
            $field_name:ident: $field_type:ty
        ),* $(,)?
    }) => {
        $vis struct $name {
            $(
                $(#[$meta])*
                pub(in $module) $field_name : $field_type
            ),*
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    if let Err(err) = app().await {
        tracing::error!("Simulation did not exit correctly: {err:?}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

async fn app() -> anyhow::Result<()> {
    assert!(is_x86_feature_detected!("avx"));

    let (layer, task) = tracing_loki::builder()
        .label("application", "lqft")?
        .label("env", "dev")?
        .build_url(Url::parse("http://127.0.0.1:3100").unwrap())?;

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_filter(tracing_subscriber::filter::LevelFilter::TRACE))
        .with(layer)
        .init();

    tokio::spawn(task);

    let mut sim = SystemBuilder::new()
        .with_params(ParamDesc {
            mass_squared: 1.0,
            coupling: 1.0,
        })
        // .enable_snapshot(SnapshotDesc {
        //     file: "snapshots/snapshots.h5".to_string(),
        //     ty: SnapshotType::Checkpoint,
        //     chunk_size: [8; 4],
        //     flush_method: FlushMethod::Sequential,
        // })
        .with_lattice(LatticeDesc::Create(LatticeCreateDesc {
            dimensions: [40, 20, 20, 20],
            initial_state: InitialState::RandomRange(-0.1..0.1),
            spacing: 1.0,
        }))
        .with_performance(PerformanceDesc {
            lattice_iter: LatticeIterMethod::Parallel
        })
        .with_observable::<MeanValue>()
        .with_observable::<Variance>()
        .with_acceptance(AcceptanceDesc {
            correction_interval: 20_000,
            initial_step_size: 1.0,
            desired_range: 0.3..0.5,
            correction_size: 0.05,
        })
        .with_burn_in(BurnInDesc {
            block_size: 100,
            required_ratio: 0.03,
            consecutive_passes: 5
        })
        .build()?;

    // visual::plot_lattice(0, sim.lattice())?;

    let total_sweeps = 5_000;
    sim.simulate_checkerboard(total_sweeps)?;

    // visual::plot_lattice(1, sim.lattice())?;

    // let stats = sim.stats();

    // let sweepx = (0..total_sweeps).map(|i| i as f64).collect::<Vec<_>>();
    //
    // let variance = stats
    //     .mean_history
    //     .iter()
    //     .zip(stats.meansq_history.iter())
    //     .map(|(&mean, &meansq)| meansq - mean.powf(2.0))
    //     .collect::<Vec<_>>();
    //
    // let tdata = (0..sim.lattice().dimensions()[0])
    //     .map(|v| v as f64)
    //     .collect::<Vec<_>>();
    // // let corr2 = sim.correlator2();
    //
    // let stats_time_mapped = stats
    //     .stats_time_history
    //     .iter()
    //     .map(|&t| t as f64)
    //     .collect::<Vec<_>>();
    // let sweep_time_mapped = stats
    //     .sweep_time_history
    //     .iter()
    //     .map(|&t| t as f64)
    //     .collect::<Vec<_>>();
    //
    // let desc = GraphDesc {
    //     dimensions: (3000, 3000),
    //     file: "layout.png",
    //     layout: (4, 3),
    //     burnin_time: 0,
    //     graphs: &[
    //         GraphData {
    //             caption: "Mean value",
    //             xdata: &sweepx,
    //             ydata: &stats.mean_history,
    //             ..Default::default()
    //         },
    //         GraphData {
    //             caption: "Variance",
    //             xdata: &sweepx,
    //             ydata: &variance,
    //             ..Default::default()
    //         },
    //         GraphData {
    //             caption: "Mean squared",
    //             xdata: &sweepx,
    //             ydata: &stats.meansq_history,
    //             ..Default::default()
    //         },
    //         GraphData {
    //             caption: "Action block average",
    //             xdata: &sweepx[sim.th_block_size() * 2..],
    //             ydata: &stats.thermalisation_ratio_history,
    //             ..Default::default()
    //         },
    //         GraphData {
    //             caption: "Action",
    //             xdata: &sweepx,
    //             ydata: &stats.action_history,
    //             ..Default::default()
    //         },
    //         GraphData {
    //             caption: "Step size",
    //             xdata: &sweepx,
    //             ydata: &stats.step_size_history,
    //             ylim: 0.75..1.25,
    //             ..Default::default()
    //         },
    //         GraphData {
    //             caption: "Acceptance ratio",
    //             xdata: &sweepx,
    //             ydata: &stats.accept_ratio_history,
    //             ylim: 20.0..80.0,
    //             ..Default::default()
    //         },
    //         GraphData {
    //             caption: "Statistics capture time",
    //             xdata: &sweepx,
    //             ydata: &stats_time_mapped,
    //             ..Default::default()
    //         },
    //         GraphData {
    //             caption: "Sweep process time",
    //             xdata: &sweepx,
    //             ydata: &sweep_time_mapped,
    //             ..Default::default()
    //         }, // GraphData {
    //            //     caption: "2-point correlator",
    //            //     xdata: &tdata,
    //            //     ydata: sim.correlator2(),
    //            //     ..Default::default()
    //            // }, // FIXME: Correlator2 seems to be filled with NaNs
    //            // GraphData {
    //            //     caption: "2-point Function",
    //            //     xdata: &tdata,
    //            //     ydata: &corr2,
    //            //     ..Default::default()
    //            // }
    //     ],
    // };
    // plot_observable(desc, &sim)?;

    sim.push_metrics();

    tracing::info!(
        "System thermalised at sweep {:?}",
        sim.stats().thermalised_at
    );

    Ok(())
}
