use std::ops::Range;
use plotters::prelude::*;
use crate::lattice::ScalarLattice4D;
use crate::sim::System;

pub fn plot_lattice(index: usize, lattice: &ScalarLattice4D) -> anyhow::Result<()> {
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

pub fn plot_observable(desc: GraphDesc, sim: &System) -> anyhow::Result<()> {
    let filename = format!("plots/{}", desc.file);
    let stats = sim.stats();

    let root = BitMapBackend::new(&filename, (desc.dimensions.0, desc.dimensions.1)).into_drawing_area();
    root.fill(&WHITE)?;

    let acceptance_range = &sim.acceptance_desc.desired_range;
    let lines = [
        format!("Lattice spacing: {:.2}", sim.lattice().spacing()),
        format!("Mass squared: {:.2}", sim.mass_squared()),
        format!("Coupling: {:.2}", sim.coupling()),
        format!("Acceptance ratio target: {:.0}%-{:.0}%", acceptance_range.start * 100.0, acceptance_range.end * 100.0),
        format!("Action block size: {} sweeps", sim.th_block_size()),
        format!("Action block average threshold {}", sim.th_threshold()),
        format!("Step size correction every {} iterations", sim.acceptance_desc.correction_interval),
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
