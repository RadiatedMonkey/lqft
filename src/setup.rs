//! Functionality related to initialisation of the system

use std::{ops::Range, sync::atomic::Ordering, time::UNIX_EPOCH};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use atomic_float::AtomicF64;
use hdf5_metno as hdf5;

use crate::{lattice::ScalarLattice4D, sim::System};
use crate::sim::{SystemSettings};
use crate::snapshot::{FlushMethod, SnapshotState};
use crate::stats::SystemStats;

impl System {
    /// Determines whether thermalisation of the system has finished.
    /// 
    /// This is done by averaging the action of a block of the last `th_block_size` sweeps
    /// and another block before that. If the relative difference `(A - B) / A` is less than
    /// `th_threshold`, the system will be marked as thermalised.
    /// 
    /// This function returns the current ratio and whether this ratio is considered thermalised.
    pub fn th_ratio(&self) -> (f64, bool) {
        let bsize = self.th_block_size();

        let action_history = &self.stats().action_history;
        let ah_len = action_history.len();
        if ah_len < 2 * bsize {
            return (0.0, false)
        }

        let last50 = &action_history[(ah_len - bsize)..];
        let l50_avg = last50.iter().copied().sum::<f64>().abs() / bsize as f64;

        let prev50 = &action_history[(ah_len - 2 * bsize)..(ah_len - bsize)];
        let p50_avg: f64 = prev50.iter().copied().sum::<f64>().abs() / bsize as f64;

        let ratio = (l50_avg - p50_avg).abs() / l50_avg;
        (ratio, ratio < self.th_threshold())
    }

    /// Check whether the current field variation is correct, otherwise adjusts it slightly.
    pub fn correct_step_size(&self) {
        debug_assert!(self.stats().thermalised_at.is_none(), "Step size should not be adjusted after the system has thermalised");

        let acceptance_ratio = self.stats.accepted_moves() as f64 / self.stats.total_moves() as f64;

        // Adjust dvar if acceptance ratio is 5% away from desired ratio
        if acceptance_ratio < self.lower_acceptance {
            let correction = 1.0 - self.step_size_correction();
            let _ = self
                .step_size
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |f| Some(f * correction));
        } else if acceptance_ratio > self.upper_acceptance {
            let correction = 1.0 + self.step_size_correction();
            let _ = self
                .step_size
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |f| Some(f * correction));
        }
    }
}

// ##########################################################################################################
// SYSTEMBUILDER
// ##########################################################################################################

/// The method of initialising the lattice.
pub enum InitialState {
    /// Sets every lattice site to the same fixed value.
    Fixed(f64),
    /// Randomises the lattice using values from this range.
    RandomRange(Range<f64>),
}

/// Used to configure a lattice simulation.
pub struct SystemBuilder {
    spacing: f64,
    sizes: [usize; 4],
    initial_state: InitialState,

    initial_step_size: f64,
    lower_acceptance: f64,
    upper_acceptance: f64,
    step_size_correction: f64,

    acceptance_update_interval: usize,
    thermalisation_threshold: f64,
    thermalisation_block_size: usize,

    snapshot_filename: Option<String>,
    snapshot_chunk_size: [usize; 5],
    snapshot_batch_size: usize,
    snapshot_flush_method: FlushMethod,

    mass_squared: f64,
    bare_coupling: f64,
}

impl SystemBuilder {
    pub fn new() -> Self {
        Self {
            snapshot_filename: None,
            snapshot_chunk_size: [1, 8, 8, 8, 8],
            snapshot_batch_size: 10,
            spacing: 1.0,
            sizes: [40, 20, 20, 20],
            initial_state: InitialState::Fixed(0.0),
            initial_step_size: 1.0,
            mass_squared: 1.0,
            bare_coupling: 0.0,
            lower_acceptance: 0.3,
            upper_acceptance: 0.5,
            step_size_correction: 0.05,
            thermalisation_threshold: 0.01,
            thermalisation_block_size: 100,
            acceptance_update_interval: 1000,
            snapshot_flush_method: FlushMethod::Sequential,
        }
    }

    pub fn snapshot_filename<T: Into<String>>(mut self, filename: T) -> Self {
        self.snapshot_filename = Some(filename.into());
        self
    }

    pub fn snapshot_chunk_size<T: Into<[usize; 5]>>(mut self, size: T) -> Self {
        self.snapshot_chunk_size = size.into();
        self
    }

    pub fn snapshot_batch_size(mut self, size: usize) -> Self {
        self.snapshot_batch_size = size;
        self
    }

    /// Sets the change in step size for each correction.
    /// 
    /// If the system determines the acceptance ratio to be outside of the desired range,
    /// it may increase or decrease the step size by this ratio.
    /// 
    /// Default value: `0.05`
    pub fn step_size_correction(mut self, value: f64) -> Self {
        self.step_size_correction = value;
        self
    }

    /// Sets the lattice spacing.
    /// 
    /// Default value: `1.0`.
    pub fn spacing(mut self, value: f64) -> Self {
        self.spacing = value;
        self
    }

    /// Sets the dimensions of the lattice.
    /// 
    /// These dimensions are in the format `[t, x, y, z]`.
    /// 
    /// Default value: `[40, 20, 20, 20]`
    pub fn sizes(mut self, value: [usize; 4]) -> Self {
        self.sizes = value;
        self
    }

    /// Sets the initial state of the lattice.
    /// 
    /// The initial state can either be set to a constant value (for example `0.0` for a "cold start")
    /// or a random range for a "hot start".
    /// 
    /// Default value: `LatticeInitialization::FixedValue(0.0)`.
    pub fn initial_value(mut self, value: InitialState) -> Self {
        self.initial_state = value;
        self
    }

    /// The starting value of the step size.
    /// 
    /// This should be set around the optimal step size for the current configuration (if known).
    /// If the initial step size is suboptimal, the system will automatically adjust it to reach the desired
    /// acceptance ratio. This may however cause thermalisation to take longer.
    /// 
    /// Default value: `1.0`.
    pub fn initial_step_size(mut self, value: f64) -> Self {
        self.initial_step_size = value;
        self
    }

    /// Sets the desired lower bound on the acceptance ratio. If the acceptance ratio goes below
    /// this bound, the step size will be adjusted to correct this.
    ///
    /// Default value: `0.3`.
    /// 
    /// See [`step_size_decrease`](SimBuilder::step_size_decrease) and [`initial_step_size`](SimBuilder::initial_step_size) for more information.
    pub fn lower_acceptance(mut self, ratio: f64) -> Self {
        self.lower_acceptance = ratio;
        self
    }

    /// Sets the desired upper bound on the acceptance ratio. If the acceptance ratio goes above
    /// this bound, the step size will be adjusted to correct this.
    /// 
    /// Default value: `0.5`.
    /// 
    /// See [`step_size_increase`](SimBuilder::step_size_increase) and [`initial_step_size`](SimBuilder::initial_step_size) for more information.
    pub fn upper_acceptance(mut self, ratio: f64) -> Self {
        self.upper_acceptance = ratio;
        self
    }

    /// Sets the amount of iterations before checking and correcting the step size.
    /// 
    /// Rather than correcting the step size every single iteration, the system checks it on a predetermined interval.
    /// This is to improve performance since most times only a few corrections are needed to steer the system to the right acceptance ratio.
    /// 
    /// Default value: `1000`.
    pub fn step_size_correction_interval(mut self, interval: usize) -> Self {
        self.acceptance_update_interval = interval;
        self
    }

    /// Sets the coupling constant.
    /// 
    /// This sets the strength of the 4phi coupling. If set to `0.0` the system will resemble a free scalar field.
    /// 
    /// Default value: `0.0`.
    pub fn coupling(mut self, value: f64) -> Self {
        self.bare_coupling = value;
        self
    }

    /// Sets the mass squared. Setting this to a negative value will introduce symmetry breaking.
    /// 
    /// Default value: `1.0`.
    pub fn mass_squared(mut self, value: f64) -> Self {
        self.mass_squared = value;
        self
    }   

    /// Sets the thermalisation threshold. See [`th_ratio`](System::th_ratio) for more information on how thermalisation
    /// works.
    /// 
    /// Default value: `0.01`.
    pub fn th_threshold(mut self, value: f64) -> Self {
        self.thermalisation_threshold = value;
        self
    }

    /// Sets the thermalisation block size. See [`th_ratio`](System::th_ratio) for more information on how thermalisation
    /// works.
    /// 
    /// Default value: `100`.
    pub fn th_block_size(mut self, value: usize) -> Self {
        self.thermalisation_block_size = value;
        self
    }

    pub fn snapshot_flush_method(mut self, method: FlushMethod) -> Self {
        self.snapshot_flush_method = method;
        self
    }

    /// Creates the simulation using the given options.
    pub fn build(self) -> anyhow::Result<System> {
        let lattice = match self.initial_state {
            InitialState::Fixed(val) => ScalarLattice4D::filled(self.sizes, val),
            InitialState::RandomRange(range) => ScalarLattice4D::random(self.sizes, range),
        };

        let snapshot = self.snapshot_filename.map(hdf5::File::append).transpose()?;
        let dataset = snapshot.as_ref().map(|archive| {
            let elapsed = UNIX_EPOCH.elapsed()?.as_secs();

            let [st, sx, sy, sz] = lattice.dimensions();

            let settings = SystemSettings {
                coupling: self.bare_coupling,
                mass_squared: self.mass_squared,
                spacing: self.spacing,
                step_size: self.initial_step_size
            };

            let set_name = format!("data-{elapsed}");
            let set = archive
                .new_dataset::<f64>()
                .chunk(self.snapshot_chunk_size)
                .shape((1.., st, sx, sy, sz))
                .shuffle()
                .fletcher32()
                .deflate(5)
                .create(set_name.as_str())?;

            let attr = set
                .new_attr::<SystemSettings>()
                .create("config")?;

            attr.write_scalar(&settings)?;

            println!("Dataset {set_name} created");

            Ok::<hdf5_metno::Dataset, anyhow::Error>(set)
        }).transpose()?;

        let snapshot_state = snapshot.map(|file| {
            SnapshotState {
                file, dataset: Arc::new(dataset.unwrap()),
                chunk_size: self.snapshot_chunk_size,
                batch_size: self.snapshot_batch_size,
                tx: None, thread: None,
                flush_method: self.snapshot_flush_method,
            }
        });

        let mut sim = System {
            simulating: AtomicBool::new(false),
            current_sweep: 0,
            lattice,
            spacing: self.spacing,
            step_size: AtomicF64::new(self.initial_step_size),
            mass_squared: self.mass_squared,
            coupling: self.bare_coupling,
            lower_acceptance: self.lower_acceptance,
            upper_acceptance: self.upper_acceptance,
            step_size_correction: self.step_size_correction,
            stats: SystemStats::default(),
            step_size_correction_interval: self.acceptance_update_interval,
            th_block_size: self.thermalisation_block_size,
            th_threshold: self.thermalisation_threshold,
            correlation_slices: Vec::new(),
            measurement_interval: 50,
            snapshot_state
        };

        let first_action = sim.compute_full_action();
        sim.stats
            .current_action
            .store(first_action, Ordering::Release);

        sim.stats.action_history.push(first_action);

        sim.stats.snapshot_batch.reserve(self.snapshot_batch_size);

        Ok(sim)
    }
}

impl Default for SystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}