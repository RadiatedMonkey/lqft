mod lattice;
mod sim;
mod vis;
mod setup;

use crate::lattice::ScalarLattice4D;
use crate::sim::{InitialState, System, SystemBuilder, SystemStats};
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};
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
        .build_cartesian_2d(0..lattice.dimensions()[2], 0..lattice.dimensions()[3])?;

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

    let grid = (0..lattice.dimensions()[3]).flat_map(|y| (0..lattice.dimensions()[2]).map(move |x| (x, y)));
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
    pub ydesc: &'a str,
    pub xdata: &'a [f64],
    pub ydata: &'a [f64],
    pub xlim: Range<f64>,
    pub ylim: Range<f64>,
    pub description: &'a str
}

impl<'a> Default for GraphData<'a> {
    fn default() -> Self {
        Self {
            caption: "",
            ydesc: "",
            xdata: &[],
            ydata: &[],
            xlim: 0.0..0.0,
            ylim: 0.0..0.0,
            description: ""
        }
    }
}

pub struct GraphDesc<'a, 'b> {
    pub dimensions: (u32, u32),
    pub file: &'a str,
    pub graphs: &'a [GraphData<'b>],
    pub layout: (usize, usize),
    /// The amount of samples to throw away. The initial samples usually generate extremely
    /// tall peaks obscuring the rest of the data.
    pub burnin_time: usize
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

fn plot_observable(desc: GraphDesc, sim: &System) -> anyhow::Result<()> {
    let filename = format!("plots/{}", desc.file);
    let stats = sim.stats();

    let root = BitMapBackend::new(&filename, (desc.dimensions.0, desc.dimensions.1)).into_drawing_area();
    root.fill(&WHITE)?;

    let lines = [
        format!("Lattice spacing: {:.2}", sim.spacing()),
        format!("Mass squared: {:.2}", sim.mass_squared()),
        format!("Coupling: {:.2}", sim.coupling()),
        format!("Acceptance ratio target: {:.0}%-{:.0}%", sim.lower_acceptance() * 100.0, sim.upper_acceptance() * 100.0),
        format!("Action block size: {} sweeps", sim.th_block_size()),
        format!("Action block average threshold {}", sim.th_threshold()),
        format!("Step size correction every {} iterations", sim.step_size_correction_interval()),
        format!("A sweep is {} iterations", sim.lattice().sweep_size()),
        format!("System reached equilibrium at sweep {}", stats.thermalised_at.map(|s| s.to_string()).unwrap_or("NA".to_owned())),
        format!("Performed {} measurements", stats.performed_measurements)
    ];

    for (i, line) in lines.iter().enumerate() {
        root.draw_text(
            line, &TextStyle::from(("sans-serif", 30)),
            (50, 50 + 30 * i as i32)
        )?;
    }

    let areas = root.split_evenly(desc.layout);

    // Skip first plot to use as text.
    for (i, area) in areas.iter().enumerate().skip(1) {
        let opt = desc.graphs.get(i - 1);
        if opt.is_none() {
            break;
        }

        let graph = opt.unwrap();
        
        let xdata = &graph.xdata[desc.burnin_time..];
        let (xmin, xmax) = if graph.xlim.is_empty() {
            (find_min(xdata), find_max(xdata))
        } else {
            (graph.xlim.start, graph.xlim.end)
        };

        let ydata = &graph.ydata[desc.burnin_time..];
        let (ymin, ymax) = if graph.ylim.is_empty() {
            (find_min(ydata), find_max(ydata))
        } else {
            (graph.ylim.start, graph.ylim.end)
        };

        let mut cc = ChartBuilder::on(area)
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .caption(graph.caption, ("sans-serif", 40))
            .build_cartesian_2d(xmin..xmax, ymin..ymax)?;

        cc.configure_mesh()
            .x_labels(10)
            .x_label_formatter(&|v| format!("{v:.0}"))
            .x_desc("Sweeps")
            .y_labels(10)
            .y_desc(graph.ydesc)
            .draw()?;

        // Plot observable
        cc.draw_series(LineSeries::new(
            xdata.iter().copied().zip(ydata.iter().copied()),
            &BLUE,
        ))?;

        if let Some(point) = stats.thermalised_at {
            // Plot thermalisation point
            cc.draw_series(DashedLineSeries::new(
                vec![(point as f64, ymin), (point as f64, ymax)],
                2, 4,
                ShapeStyle {
                    color: RED.mix(1.0),
                    filled: false,
                    stroke_width: 1
                }
            ))?;
        }   
    }

    root.present()?;
    println!("Graph has been plotted in {filename}");

    Ok(())
}

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
        .thermalisation_block_size(100)
        .thermalisation_threshold(0.005)
        .build()?;

    plot_lattice(0, sim.lattice())?;

    let total_sweeps = 50000;
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
