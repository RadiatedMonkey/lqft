mod lattice;

use plotters::prelude::*;
use crate::lattice::Lattice4D;

const OUT_FILE_NAME: &str = "plots/plot.png";

fn timestep() {

}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lattice = lattice::Lattice4D::<50>::random();

    let root = BitMapBackend::new(OUT_FILE_NAME, (750, 750)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Lattice simulation", ("sans-serif", 80))
        .margin(5)
        .top_x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0usize..50, 0usize..50)?;

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
        (0..50usize.pow(2)).into_iter().map(|i| {
            let [_, _, y, z] = Lattice4D::<50>::from_index(i);
            Rectangle::new(
                [(y, z), (y + 1, z + 1)],
                HSLColor(
                    240.0 / 360.0 - 240.0 / 360.0 * (lattice[i] / 20.0),
                    0.7,
                    0.1 + 0.4 * lattice[i] / 20.0
                ).filled()
            )
        })
    )?;

    root.present().expect("Unable to write to file");
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_to_coords_test() {
        use lattice::*;

        const N: usize = 50;
        type Lat = Lattice4D<N>;

        for i in 0..N.pow(4) {
            let coords = Lat::from_index(i);
            let idx = Lat::to_index(coords);

            println!("{coords:?}");
            assert_eq!(i, idx, "Conversion between indices and coordinates is incorrect!");
        }
    }
}