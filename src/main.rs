mod lattice;
mod sim;

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

    root.present().expect("Unable to write to file");
    println!("Result has been saved to {filename}");

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut sim = SimBuilder::new()
        .sizes([25; 4])
        .spacing(1.0)
        .variation(1.0)
        .mass_squared(1.0)
        .bare_coupling(0.5)
        .initial_value(InitialFieldValue::Fixed(0.0))
        .build()?;

    let sweep = sim.lattice().sweep_size();

    plot_lattice(0, sim.lattice())?;
    let mut last_update = Instant::now();

    let mut means = Vec::new();
    let mut vars = Vec::new();

    let total_count = 50 * sweep;
    for i in 0..total_count {
        sim.timestep();

        if i % (sweep / 10) == 0 {
            let mean = sim.lattice().mean();
            let var = sim.lattice().variance();

            println!("Mean: {mean}, variance: {var}");

            means.push(mean);
            vars.push(var);
        }

        if last_update.elapsed().as_millis() >= 1000 {
            let accept = sim.accepted_moves();
            let total = sim.total_moves();
            let ratio = accept as f64 / total as f64 * 100.0;
            println!("---------------------------------------------");
            println!("Progress ({:.2}%): {i}/{total_count}", i as f64 / total_count as f64 * 100.0);
            println!("Acceptance ratio: {:.2}%", ratio);
            // println!("Mean field value: {}", sim.lattice().avg());

            // plot_lattice(i, sim.lattice())?;
            last_update = Instant::now();
        }
    }

    let root = BitMapBackend::new("plots/avg.png", (1024, 768)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(
            "Mean field value over time",
            ("sans-serif", 40),
        )
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Right, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(
            0..means.len(),
            means.iter().copied().reduce(|a, b| if a.total_cmp(&b).is_lt() { a } else { b })
                .unwrap_or(0.0)..means.iter().copied().reduce(|a, b| if a.total_cmp(&b).is_gt() { a } else { b })
                .unwrap_or(0.0)
        )?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(30)
        .max_light_lines(4)
        .y_desc("Mean field value")
        .draw()?;

    println!("avg len: {}", means.len());

    chart.draw_series(LineSeries::new(
        means.iter().enumerate().map(|(i, v)| (i, *v)),
        &BLUE,
    ))?;

    root.present()?;
    println!("Average plotted over time");

    let root = BitMapBackend::new("plots/var.png", (1024, 768)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(
            "Field value variance over time",
            ("sans-serif", 40),
        )
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Right, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(
            0..vars.len(),
            vars.iter().copied().reduce(|a, b| if a.total_cmp(&b).is_lt() { a } else { b })
                .unwrap_or(0.0)..vars.iter().copied().reduce(|a, b| if a.total_cmp(&b).is_gt() { a } else { b })
                .unwrap_or(0.0)
        )?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(30)
        .max_light_lines(4)
        .y_desc("Variance")
        .draw()?;

    chart.draw_series(LineSeries::new(
        vars.iter().enumerate().map(|(i, v)| (i, *v)),
        &RED,
    ))?;

    root.present()?;
    println!("Variance plotted over time");

    Ok(())
}