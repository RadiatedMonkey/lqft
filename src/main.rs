mod lattice;
mod sim;
mod vis;

use std::ops::Range;
use std::time::Instant;
use plotters::prelude::*;
use crate::lattice::ScalarLattice4D;
use crate::sim::{InitialFieldValue, SimBuilder};

fn plot_lattice(index: usize, lattice: &ScalarLattice4D) -> anyhow::Result<()> {
    let filename = format!("plots/step-{index}.png");

    let root = BitMapBackend::new(&filename, (750, 750)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Lattice simulation (avg: {})", lattice.mean()), ("sans-serif", 80))
        .margin(5)
        .top_x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0usize..25, 0usize..25)?;

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

    chart.draw_series(
        (0..25usize.pow(2)).into_iter().map(|i| {
            let [_, _, y, z] = lattice.from_index(i);
            let val = lattice[i];

            // Scale val from [-max, max] to [0, 1] for the color mapping
            // Assuming max fluctuation is around 2.0 based on your previous delta
            let max_scale = 2.0;
            let normalized = (val / max_scale).clamp(-1.0, 1.0);

            let color = if normalized > 0.0 {
                // Positive: White to Red
                RGBColor(255, (255.0 * (1.0 - normalized)) as u8, (255.0 * (1.0 - normalized)) as u8)
            } else {
                // Negative: White to Blue
                let n = normalized.abs();
                RGBColor((255.0 * (1.0 - n)) as u8, (255.0 * (1.0 - n)) as u8, 255)
            };

            Rectangle::new([(y, z), (y + 1, z + 1)], color.filled())
        })
    )?;

    root.present()?;
    println!("Result has been saved to {filename}");

    Ok(())
}

pub struct GraphData<'a> {
    pub caption: &'a str,
    pub xdata: &'a [f64],
    pub ydata: &'a [f64],
    pub xlim: Range<f64>,
    pub ylim: Range<f64>
}

impl<'a> Default for GraphData<'a> {
    fn default() -> Self {
        Self {
            caption: "",
            xdata: &[],
            ydata: &[],
            xlim: 0.0..0.0,
            ylim: 0.0..0.0
        }
    }
}

pub struct GraphDesc<'a, 'b> {
    pub file: &'a str,
    pub graphs: &'a [GraphData<'b>],
    pub layout: (usize, usize)
}

/// Custom function to find the smallest float in a slice since floats don't implement Eq.
fn find_min(data: &[f64]) -> f64 {
    data.iter().copied().reduce(|a, b| {
        if a.total_cmp(&b).is_lt() { a } else { b }
    }).unwrap_or(0.0)
}

/// Custom function to find the largest float in a slice since floats don't implement Eq.
fn find_max(data: &[f64]) -> f64 {
    data.iter().copied().reduce(|a, b| {
        if a.total_cmp(&b).is_gt() { a } else { b }
    }).unwrap_or(0.0)
}

fn plot_function(desc: GraphDesc) -> anyhow::Result<()> {
    let filename = format!("plots/{}", desc.file);

    let root = BitMapBackend::new(&filename, (2000, 3000)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let areas = root.split_evenly(desc.layout);
    for (i, area) in areas.iter().enumerate() {
        let opt = desc.graphs.get(i);
        if opt.is_none() {
            break
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
        
        cc.configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .draw()?;
        
        cc.draw_series(LineSeries::new(
            graph.xdata.iter().copied().zip(graph.ydata.iter().copied()),
            &BLUE
        ))?;
    }

    root.present()?;
    println!("Graph has been plotted in {filename}");

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut sim = SimBuilder::new()
        .sizes([50, 20, 20, 20])
        .spacing(1.0)
        .initial_step_size(10.0)
        .upper_acceptance(0.4)
        .lower_acceptance(0.3)
        .mass_squared(1.0)
        .stats_interval(100000)
        .coupling(0.1)
        .initial_value(InitialFieldValue::RandomRange(-0.1..0.1))
        .build()?;

    sim.simulate_sequential(50);

    // plot_lattice(0, sim.lattice())?;

    let stats = sim.stats();
    let sweep_size = sim.lattice().sweep_size();

    let accepted_moves = stats.accepted_move_history.iter().map(|v| *v as f64).collect::<Vec<_>>();
    let sweepx = stats.timepoints.iter().map(|t| (*t as f64) / (sweep_size as f64)).collect::<Vec<_>>();

    let tdata = (0..sim.lattice().sizes()[0]).map(|v| v as f64).collect::<Vec<_>>();
    let corr2 = sim.correlator2();

    let desc = GraphDesc {
        file: "layout.png",
        layout: (3, 2),
        graphs: &[
            GraphData {
                caption: "Field Mean",
                xdata: &sweepx,
                ydata: &stats.mean_history,
                ..Default::default()
            },
            GraphData {
                caption: "Field Mean Squared",
                xdata: &sweepx,
                ydata: &stats.var_history,
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
                ylim: 0.0..(stats.total_moves as f64),
                ..Default::default()
            },
            GraphData {
                caption: "2-point Function",
                xdata: &tdata,
                ydata: &corr2,
                ..Default::default()
            }
        ]
    };
    plot_function(desc)?;

    // let root = BitMapBackend::new("plots/avg.png", (1024, 768)).into_drawing_area();

    // root.fill(&WHITE)?;

    // let means = &sim.stats().mean_history;
    // let mut chart = ChartBuilder::on(&root)
    //     .margin(10)
    //     .caption(
    //         "Mean field value over time",
    //         ("sans-serif", 40),
    //     )
    //     .set_label_area_size(LabelAreaPosition::Left, 60)
    //     .set_label_area_size(LabelAreaPosition::Right, 60)
    //     .set_label_area_size(LabelAreaPosition::Bottom, 40)
    //     .build_cartesian_2d(
    //         0..means.len(),
    //         means.iter().copied().reduce(|a, b| if a.total_cmp(&b).is_lt() { a } else { b })
    //             .unwrap_or(0.0)..means.iter().copied().reduce(|a, b| if a.total_cmp(&b).is_gt() { a } else { b })
    //             .unwrap_or(0.0)
    //     )?;

    // chart
    //     .configure_mesh()
    //     .disable_x_mesh()
    //     .disable_y_mesh()
    //     .x_labels(30)
    //     .max_light_lines(4)
    //     .y_desc("Mean field value")
    //     .draw()?;

    // println!("avg len: {}", means.len());

    // chart.draw_series(LineSeries::new(
    //     means.iter().enumerate().map(|(i, v)| (i, *v)),
    //     &BLUE,
    // ))?;

    // root.present()?;
    // println!("Average plotted over time");

    // let root = BitMapBackend::new("plots/var.png", (1024, 768)).into_drawing_area();

    // root.fill(&WHITE)?;

    // let vars = &sim.stats().var_history;
    // let mut chart = ChartBuilder::on(&root)
    //     .margin(10)
    //     .caption(
    //         "Field value variance over time",
    //         ("sans-serif", 40),
    //     )
    //     .set_label_area_size(LabelAreaPosition::Left, 60)
    //     .set_label_area_size(LabelAreaPosition::Right, 60)
    //     .set_label_area_size(LabelAreaPosition::Bottom, 40)
    //     .build_cartesian_2d(
    //         0..vars.len(),
    //         vars.iter().copied().reduce(|a, b| if a.total_cmp(&b).is_lt() { a } else { b })
    //             .unwrap_or(0.0)..vars.iter().copied().reduce(|a, b| if a.total_cmp(&b).is_gt() { a } else { b })
    //             .unwrap_or(0.0)
    //     )?;

    // chart
    //     .configure_mesh()
    //     .disable_x_mesh()
    //     .disable_y_mesh()
    //     .x_labels(30)
    //     .max_light_lines(4)
    //     .y_desc("Variance")
    //     .draw()?;

    // chart.draw_series(LineSeries::new(
    //     vars.iter().enumerate().map(|(i, v)| (i, *v)),
    //     &BLUE,
    // ))?;

    // root.present()?;
    // println!("Variance plotted over time");

    // let root = BitMapBackend::new("plots/accept.png", (1024, 768)).into_drawing_area();

    // root.fill(&WHITE)?;

    // let accept_history = &sim.stats().accept_ratio_history;
    // let mut chart = ChartBuilder::on(&root)
    //     .margin(10)
    //     .caption(
    //         "Acceptance ratio over time",
    //         ("sans-serif", 40),
    //     )
    //     .set_label_area_size(LabelAreaPosition::Left, 60)
    //     .set_label_area_size(LabelAreaPosition::Right, 60)
    //     .set_label_area_size(LabelAreaPosition::Bottom, 40)
    //     .build_cartesian_2d(
    //         0..accept_history.len(),
    //         accept_history.iter().copied().reduce(|a, b| if a.total_cmp(&b).is_lt() { a } else { b })
    //             .unwrap_or(0.0)..accept_history.iter().copied().reduce(|a, b| if a.total_cmp(&b).is_gt() { a } else { b })
    //             .unwrap_or(0.0)
    //     )?;

    // chart
    //     .configure_mesh()
    //     .disable_x_mesh()
    //     .disable_y_mesh()
    //     .x_labels(30)
    //     .max_light_lines(4)
    //     .y_desc("Acceptance ratio")
    //     .draw()?;

    // chart.draw_series(LineSeries::new(
    //     accept_history.iter().enumerate().map(|(i, v)| (i, *v)),
    //     &BLUE,
    // ))?;

    plot_lattice(0, sim.lattice())?;

    Ok(())
}