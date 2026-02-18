use std::net::SocketAddr;
use crate::lattice::Lattice;
use crate::sim::System;
use plotters::prelude::*;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;
use num_traits::Pow;
use crate::stats::SystemStats;

pub fn plot_lattice(index: usize, lattice: &Lattice) -> anyhow::Result<()> {
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

    let grid = (0..lattice.dimensions()[3])
        .flat_map(|y| (0..lattice.dimensions()[2]).map(move |x| (x, y)));
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
    tracing::info!("Result has been saved to {filename}");

    Ok(())
}

pub struct GraphData<'a> {
    pub caption: &'a str,
    pub ydesc: &'a str,
    pub xdata: &'a [f64],
    pub ydata: &'a [f64],
    pub xlim: Range<f64>,
    pub ylim: Range<f64>,
    pub description: &'a str,
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
            description: "",
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
    pub burnin_time: usize,
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

    let root =
        BitMapBackend::new(&filename, (desc.dimensions.0, desc.dimensions.1)).into_drawing_area();
    root.fill(&WHITE)?;

    let avg_stats_capture_time = stats
        .stats_time_history
        .iter()
        .map(|&t| t as f64)
        .sum::<f64>()
        / stats.stats_time_history.len() as f64;
    let avg_sweep_time = stats
        .sweep_time_history
        .iter()
        .map(|&t| t as f64)
        .sum::<f64>()
        / stats.sweep_time_history.len() as f64;

    let acceptance_range = &sim.acceptance_desc.desired_range;
    let lines = [
        format!("Lattice spacing: {:.2}", sim.lattice().spacing()),
        format!("Mass squared: {:.2}", sim.mass_squared()),
        format!("Coupling: {:.2}", sim.coupling()),
        format!(
            "Acceptance ratio target: {:.0}%-{:.0}%",
            acceptance_range.start * 100.0,
            acceptance_range.end * 100.0
        ),
        format!("Action block size: {} sweeps", sim.th_block_size()),
        format!("Action block average threshold {}", sim.th_threshold()),
        format!(
            "Step size correction every {} iterations",
            sim.acceptance_desc.correction_interval
        ),
        format!("A sweep is {} iterations", sim.lattice().sweep_size()),
        format!(
            "System reached equilibrium at sweep {}",
            stats
                .thermalised_at
                .map(|s| s.to_string())
                .unwrap_or("NA".to_owned())
        ),
        format!("Performed {} measurements", stats.performed_measurements),
        format!("Average statistics capture time: {avg_stats_capture_time} microseconds"),
        format!("Average sweep process time: {avg_sweep_time} microseconds"),
    ];

    for (i, line) in lines.iter().enumerate() {
        root.draw_text(
            line,
            &TextStyle::from(("sans-serif", 30)),
            (50, 50 + 30 * i as i32),
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
                2,
                4,
                ShapeStyle {
                    color: RED.mix(1.0),
                    filled: false,
                    stroke_width: 1,
                },
            ))?;
        }
    }

    root.present()?;
    tracing::info!("Graph has been plotted in {filename}");

    Ok(())
}

use prometheus::Encoder;
use prometheus_exporter::Exporter;
use prometheus_exporter::prometheus as ps;

pub fn set_int_to(ctr: &ps::IntCounter, new_value: u64) {
    let inc = new_value - ctr.get();
    ctr.inc_by(inc);
}

pub struct MetricState {
    total_moves: ps::IntCounter,
    accepted_moves: ps::IntCounter,
    accept_ratio: ps::Gauge,
    progress: ps::Gauge,

    step_size: ps::Gauge,
    action: ps::Gauge,
    performed_measurements: ps::IntCounter,

    sweep_time: ps::Gauge,
    stats_time: ps::Gauge,
    therm_ratio: ps::Gauge,
    completed_sweeps: ps::IntCounter,
    thermalised_at: ps::IntCounter,

    mean: ps::Gauge,
    meansq: ps::Gauge,
    var: ps::Gauge,

    cpu: ps::Gauge,
    mem: ps::Gauge,

    sys: sysinfo::System,
    pid: sysinfo::Pid,
    pid_ctr: ps::IntCounter,
    runtime: ps::Gauge
}

impl MetricState {
    pub fn new() -> anyhow::Result<Self> {
        let total_moves = ps::register_int_counter!("total_moves", "Total Moves")?;
        let accepted_moves = ps::register_int_counter!("accepted_moves", "Accepted Moves")?;
        let accept_ratio = ps::register_gauge!("accept_ratio", "Accept Ratio")?;
        let progress = ps::register_gauge!("progress", "Progress")?;
        let step_size = ps::register_gauge!("step_size", "Step Size")?;
        let action = ps::register_gauge!("action", "Action")?;
        let performed_measurements = ps::register_int_counter!("performed_measurements", "Performed Measurements")?;
        let sweep_time = ps::register_gauge!("sweep_time", "Sweep Time")?;
        let stats_time = ps::register_gauge!("stats_time", "Stats Time")?;
        let therm_ratio = ps::register_gauge!("therm_ratio", "Thermalisation Ratio")?;
        let completed_sweeps = ps::register_int_counter!("completed_sweeps", "Completed Sweeps")?;
        let thermalised_at = ps::register_int_counter!("thermalised_at", "Thermalised At")?;

        let mean = ps::register_gauge!("field_mean", "Mean Value")?;
        let meansq = ps::register_gauge!("field_meansq", "Mean Squared Value")?;
        let var = ps::register_gauge!("field_variance", "Variance")?;

        let cpu = ps::register_gauge!("cpu", "CPU usage")?;
        let mem = ps::register_gauge!("mem", "Memory usage")?;
        let pid_ctr = ps::register_int_counter!("pid", "PID")?;

        let pid = sysinfo::get_current_pid().unwrap();
        pid_ctr.inc_by(pid.as_u32() as u64);

        let runtime = ps::register_gauge!("run_time", "Run time")?;

        let running = Arc::new(AtomicBool::new(true));
        let clone = Arc::clone(&running);

        let addr: SocketAddr = "127.0.0.1:9184".parse()?;
        prometheus_exporter::start(addr).expect("Failed to start Prometheus exporter");

        Ok(Self {
            total_moves,
            accepted_moves,
            progress,
            step_size,
            action,
            performed_measurements,
            sweep_time,
            accept_ratio,
            therm_ratio,
            mean, meansq, var,
            stats_time,
            completed_sweeps,
            thermalised_at,
            cpu, mem, sys: sysinfo::System::new_all(),
            pid_ctr, pid,
            runtime
        })
    }
}

impl System {
    pub fn push_metrics(&mut self) {
        let metrics = &mut self.metrics;

        set_int_to(&metrics.total_moves, self.data.total_moves.load(Ordering::SeqCst));
        set_int_to(&metrics.accepted_moves, self.data.accepted_moves.load(Ordering::SeqCst));

        let mean = *self.data.mean_history.last().unwrap();
        let meansq = *self.data.meansq_history.last().unwrap();

        metrics.mean.set(mean);
        metrics.meansq.set(meansq);
        metrics.var.set(meansq - mean.pow(2));

        let accept_ratio = *self.data.accept_ratio_history.last().unwrap();
        metrics.accept_ratio.set(accept_ratio);

        let progress = self.data.current_sweep as f64 / self.data.desired_sweeps as f64;
        metrics.progress.set(progress);

        metrics.step_size.set(self.current_step_size.load(Ordering::Relaxed));
        metrics.action.set(self.data.current_action.load(Ordering::Relaxed));

        set_int_to(&metrics.performed_measurements, self.data.performed_measurements as u64);

        let sweep_time = *self.data.sweep_time_history.last().unwrap();
        metrics.sweep_time.set(sweep_time as f64);

        let stats_time = *self.data.stats_time_history.last().unwrap();
        metrics.stats_time.set(stats_time as f64);

        let therm_ratio = self.data.thermalisation_ratio_history.last().copied().unwrap_or(0.0);
        metrics.therm_ratio.set(therm_ratio);

        set_int_to(&metrics.completed_sweeps, self.data.current_sweep as u64 + 1);

        if metrics.thermalised_at.get() == 0 && let Some(sweep) = self.data.thermalised_at {
            metrics.thermalised_at.inc_by(sweep as u64);
        }

        metrics.sys.refresh_all();
        if let Some(proc) = metrics.sys.process(metrics.pid) {
            metrics.cpu.set(proc.cpu_usage() as f64);
            metrics.mem.set(proc.memory() as f64);
            metrics.runtime.set(proc.run_time() as f64);
        }
    }
}