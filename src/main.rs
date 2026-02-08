mod lattice;
mod sim;
mod vis;

use crate::lattice::ScalarLattice4D;
use crate::sim::{InitialFieldValue, SimBuilder};
use plotters::prelude::*;
use std::ops::Range;

fn plot_lattice(index: usize, lattice: &ScalarLattice4D) -> anyhow::Result<()> {
    let filename = format!("plots/step-{index}.png");

    let root = BitMapBackend::new(&filename, (750, 750)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Lattice", ("sans-serif", 80))
        .margin(5)
        .top_x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..lattice.sizes()[2], 0..lattice.sizes()[3])?;

    chart
        .configure_mesh()
        .x_labels(15)
        .y_labels(15)
        .max_light_lines(4)
        .x_label_offset(35)
        .y_label_offset(25)
        .disable_x_mesh()
        .disable_y_mesh()
        .label_style(("sans-serif", 20))
        .draw()?;

    let grid = (0..lattice.sizes()[3]).flat_map(|y| (0..lattice.sizes()[2]).map(move |x| (x, y)));
    chart.draw_series(grid.map(|(y, z)| {
        let val = unsafe { *lattice[[0, 0, y, z]].get() };

        // Scale val from [-max, max] to [0, 1] for the color mapping
        // Assuming max fluctuation is around 2.0 based on your previous delta
        let max_scale = 2.0;
        let normalized = (val / max_scale).clamp(-1.0, 1.0);

        let color = if normalized > 0.0 {
            // Positive: White to Red
            RGBColor(
                255,
                (255.0 * (1.0 - normalized)) as u8,
                (255.0 * (1.0 - normalized)) as u8,
            )
        } else {
            // Negative: White to Blue
            let n = normalized.abs();
            RGBColor((255.0 * (1.0 - n)) as u8, (255.0 * (1.0 - n)) as u8, 255)
        };

        Rectangle::new([(y, z), (y + 1, z + 1)], color.filled())
    }))?;

    root.present()?;
    println!("Result has been saved to {filename}");

    Ok(())
}

pub struct GraphData<'a> {
    pub caption: &'a str,
    pub xdata: &'a [f64],
    pub ydata: &'a [f64],
    pub xlim: Range<f64>,
    pub ylim: Range<f64>,
}

impl<'a> Default for GraphData<'a> {
    fn default() -> Self {
        Self {
            caption: "",
            xdata: &[],
            ydata: &[],
            xlim: 0.0..0.0,
            ylim: 0.0..0.0,
        }
    }
}

pub struct GraphDesc<'a, 'b> {
    pub file: &'a str,
    pub graphs: &'a [GraphData<'b>],
    pub layout: (usize, usize),
}

/// Custom function to find the smallest float in a slice since floats don't implement Eq.
fn find_min(data: &[f64]) -> f64 {
    data.iter()
        .copied()
        .reduce(|a, b| if a.total_cmp(&b).is_lt() { a } else { b })
        .unwrap_or(0.0)
}

/// Custom function to find the largest float in a slice since floats don't implement Eq.
fn find_max(data: &[f64]) -> f64 {
    data.iter()
        .copied()
        .reduce(|a, b| if a.total_cmp(&b).is_gt() { a } else { b })
        .unwrap_or(0.0)
}

fn plot_function(desc: GraphDesc) -> anyhow::Result<()> {
    let filename = format!("plots/{}", desc.file);

    let root = BitMapBackend::new(&filename, (3000, 3000)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly(desc.layout);
    for (i, area) in areas.iter().enumerate() {
        let opt = desc.graphs.get(i);
        if opt.is_none() {
            break;
        }

        let graph = opt.unwrap();

        let (xmin, xmax) = if graph.xlim.is_empty() {
            (find_min(graph.xdata), find_max(graph.xdata))
        } else {
            (graph.xlim.start, graph.xlim.end)
        };

        let (ymin, ymax) = if graph.ylim.is_empty() {
            (find_min(graph.ydata), find_max(graph.ydata))
        } else {
            (graph.ylim.start, graph.ylim.end)
        };

        let mut cc = ChartBuilder::on(area)
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .caption(graph.caption, ("sans-serif", 40))
            .build_cartesian_2d(xmin..xmax, ymin..ymax)?;

        cc.configure_mesh().x_labels(10).y_labels(10).draw()?;

        cc.draw_series(LineSeries::new(
            graph.xdata.iter().copied().zip(graph.ydata.iter().copied()),
            &BLUE,
        ))?;
    }

    root.present()?;
    println!("Graph has been plotted in {filename}");

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut sim = SimBuilder::new()
        .sizes([40, 15, 15, 15])
        .spacing(1.0)
        .initial_step_size(0.75)
        .upper_acceptance(0.5)
        .lower_acceptance(0.3)
        .mass_squared(0.5)
        .coupling(0.0)
        .initial_value(InitialFieldValue::RandomRange(-0.1..0.1))
        .thermalisation_block_size(100)
        .thermalisation_threshold(0.01)
        .build()?;

    plot_lattice(0, sim.lattice())?;

    let total_sweeps = 500;
    sim.simulate_checkerboard(total_sweeps);

    println!("Printing lattice");
    plot_lattice(1, sim.lattice())?;

    let stats = sim.stats();

    let accepted_moves = stats
        .accepted_move_history
        .iter()
        .map(|v| *v as f64)
        .collect::<Vec<_>>();
    let sweepx = (0..total_sweeps).map(|i| i as f64).collect::<Vec<_>>();

    let variance = stats
        .mean_history
        .iter()
        .zip(stats.meansq_history.iter())
        .map(|(&mean, &meansq)| meansq - mean.powf(2.0))
        .collect::<Vec<_>>();

    // let tdata = (0..sim.lattice().sizes()[0]).map(|v| v as f64).collect::<Vec<_>>();
    // let corr2 = sim.correlator2();

    let desc = GraphDesc {
        file: "layout.png",
        layout: (3, 3),
        graphs: &[
            GraphData {
                caption: "Field Mean",
                xdata: &sweepx,
                ydata: &stats.mean_history,
                ..Default::default()
            },
            GraphData {
                caption: "Field Variance",
                xdata: &sweepx,
                ydata: &variance,
                ..Default::default()
            },
            GraphData {
                caption: "Field Mean Squared",
                xdata: &sweepx,
                ydata: &stats.meansq_history,
                ..Default::default()
            },
            GraphData {
                caption: &format!("Action block average ratio (size {})", sim.thermalisation_block_size()),
                xdata: &sweepx,
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
                caption: "Step Size",
                xdata: &sweepx,
                ydata: &stats.dvar_history,
                ylim: 0.75..1.25,
                ..Default::default()
            },
            GraphData {
                caption: "Acceptance Ratio",
                xdata: &sweepx,
                ydata: &stats.accept_ratio_history,
                ylim: 20.0..80.0,
                ..Default::default()
            },
            GraphData {
                caption: "Accepted Moves",
                xdata: &sweepx,
                ydata: &accepted_moves,
                ylim: 0.0..(sim.total_moves() as f64),
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
    plot_function(desc)?;

    Ok(())
}
