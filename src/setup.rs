//! Functionality related to initialisation of the system

use atomic_float::AtomicF64;
use hdf5_metno as hdf5;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::{ops::Range, sync::atomic::Ordering, time::UNIX_EPOCH};

use crate::snapshot::SnapshotState;
use crate::stats::SystemStats;
use crate::{lattice::ScalarLattice4D, sim::System};

impl System {
    /// Determines whether thermalisation of the system has finished.
    ///
    /// This is done by averaging the action of a block of the last `th_block_size` sweeps
    /// and another block before that. If the relative difference `(A - B) / A` is less than
    /// `th_threshold`, the system will be marked as thermalised.
    ///
    /// This function returns the current ratio and whether this ratio is considered thermalised.
    pub fn compute_burn_in_ratio(&self) -> (f64, bool) {
        let bsize = self.th_block_size();

        let action_history = &self.stats().action_history;
        let ah_len = action_history.len();
        if ah_len < 2 * bsize {
            return (0.0, false);
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
        debug_assert!(
            self.stats().thermalised_at.is_none(),
            "Step size should not be adjusted after the system has thermalised"
        );

        let acceptance_ratio = self.stats.accepted_moves() as f64 / self.stats.total_moves() as f64;

        // Adjust dvar if acceptance ratio is 5% away from desired ratio
        if acceptance_ratio < self.acceptance_desc.desired_range.start {
            let correction = 1.0 - self.acceptance_desc.correction_size;
            let _ = self
                .current_step_size
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |f| Some(f * correction));
        } else if acceptance_ratio > self.acceptance_desc.desired_range.end {
            let correction = 1.0 + self.acceptance_desc.correction_size;
            let _ = self
                .current_step_size
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |f| Some(f * correction));
        }
    }
}

// ##########################################################################################################
// SYSTEMBUILDER
// ##########################################################################################################

/// Which method to use for flushing snapshots fragments.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum FlushMethod {
    /// Saves all fragments sequentially. This uses less RAM but is much slower.
    Sequential,
    /// Creates a batch of `n` sweeps and flushes them all at once.
    Batched(usize),
}

/// Configures the lattice.
#[derive(Debug, Clone)]
pub struct LatticeDesc {
    /// The initial state of the lattice.
    ///
    /// This sets the initial conditions of the system.
    ///
    /// Default value: `InitialState::RandomRange(-0.5..0.5)`.
    pub initial_state: InitialState,
    /// The dimensions of the lattice.
    ///
    /// Default value: `[40, 20, 20, 20]`.
    pub dimensions: [usize; 4],
    /// The lattice spacing.
    ///
    /// Default value: `1.0`.
    pub spacing: f64,
}

impl Default for LatticeDesc {
    fn default() -> Self {
        Self {
            initial_state: InitialState::RandomRange(-0.5..0.5),
            dimensions: [40, 20, 20, 20],
            spacing: 1.0,
        }
    }
}

/// Controls how the system reaches the desired acceptance ratio.
///
/// The acceptance ratio is the percentage of moves that are accepted.
/// While thermalising, the system will actively adjust the step size to reach the optimal
/// acceptance ratio.
#[derive(Debug, Clone)]
pub struct AcceptanceDesc {
    /// The range of acceptance ratios that the system will consider valid. If the acceptance
    /// ratio leaves this range, the step size will be adjusted to bring it back in range.
    ///
    /// Default value: `0.3..0.5`.
    pub desired_range: Range<f64>,
    /// The size of the correction to the step size. This is percentual change (e.g. `0.05` gives
    /// a 95% decrease or 105% increase in step size).
    ///
    /// Default value: `0.05`.
    pub correction_size: f64,
    /// The amount of sweeps between step size corrections.
    ///
    /// Default value: `20_000`.
    pub correction_interval: usize,
    /// The initial step size at the beginning of the simulation.
    ///
    /// If possible, set this to a value close to the optimal step size. For suboptimal step sizes
    /// the system can take long to thermalise.
    ///
    /// Default value: `1.0`.
    pub initial_step_size: f64,
}

impl Default for AcceptanceDesc {
    fn default() -> Self {
        Self {
            desired_range: 0.3..0.5,
            correction_size: 0.05,
            correction_interval: 20_000,
            initial_step_size: 1.0,
        }
    }
}

/// Configures the properties that determine whether the system has thermalised.
///
/// Thermalisation/Burn-in is important to ensure that the system is in equilibrium and
/// the measurements will be reliable.
///
/// The systems averages the action over the last `block_size` sweeps and compares this to the block
/// before that. If the relative difference is less than `desired_ratio`, the system will be
/// considered thermalised.
#[derive(Debug, Clone)]
pub struct BurnInDesc {
    /// The size of a single block.
    ///
    /// Note that a block size of `n` implies that the system will only begin checking its state
    /// after `2n` sweeps. Therefore, for long block sizes the system might wait unnecessarily
    /// long before considering itself thermalised.
    ///
    /// Default value: `100`.
    pub block_size: usize,
    /// The relative difference required for the system to be considered stabilised.
    ///
    /// Default value: `0.1`.
    pub required_ratio: f64,
}

impl Default for BurnInDesc {
    fn default() -> Self {
        Self {
            block_size: 100,
            required_ratio: 0.1,
        }
    }
}

/// The method of initialising the lattice.
#[derive(Debug, Clone, PartialEq)]
pub enum InitialState {
    /// Sets every lattice site to the same fixed value.
    Fixed(f64),
    /// Randomises the lattice using values from this range.
    RandomRange(Range<f64>),
}

/// The type of snapshots to take.
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotType {
    /// Store every `n` sweeps
    Interval(usize),
    /// Only store final sweep
    Checkpoint,
}

/// Configuration for the snapshot feature.
#[derive(Debug, Clone)]
pub struct SnapshotDesc {
    /// The HDF5 file to output snapshots into.
    ///
    /// This can be an already existing HDF5 file, in which case the content will be appended to the
    /// file.
    ///
    /// Default value: `snapshots/snapshot.h5`.
    pub file: String,
    /// The type of snapshots to take. See [`SnapshotType`] for more information.
    ///
    /// Default value: `SnapshotType::Checkpoint`.
    pub ty: SnapshotType,
    /// The chunk size of the dataspace. The dataspace in the HDF5 file is split into 5 dimensions.
    /// One for each sweep and the rest for the lattice dimensions. The chunk size configures the
    /// lattice chunking dimensions. These are parts of the lattice that will be processed independently
    /// when storing.
    ///
    /// See [`HDF5 chunking`](https://support.hdfgroup.org/documentation/hdf5-docs/advanced_topics/chunking_in_hdf5.html) for more information.
    ///
    /// Default value: `[16, 16, 16, 16]`.
    pub chunk_size: [usize; 4],
    /// The method used to flush the snapshots to disk.
    ///
    /// Default value: `FlushMethod::Sequential`.
    pub flush_method: FlushMethod,
}

impl Default for SnapshotDesc {
    fn default() -> Self {
        Self {
            file: "snapshots/snapshot.h5".to_owned(),
            ty: SnapshotType::Checkpoint,
            chunk_size: [16; 4],
            flush_method: FlushMethod::Sequential,
        }
    }
}

/// Represents the physical properties of the system.
///
/// These are the parameters of the theory such as the coupling constant and mass.
#[derive(Debug, Clone)]
pub struct ParamDesc {
    /// The coupling constant of the theory.
    pub coupling: f64,
    /// The squared mass of the field.
    pub mass_squared: f64,
}

impl Default for ParamDesc {
    fn default() -> Self {
        Self {
            coupling: 0.0,
            mass_squared: 0.0,
        }
    }
}

/// Used to configure a lattice simulation.
#[derive(Debug, Clone)]
pub struct SystemBuilder {
    param_desc: ParamDesc,
    lattice_desc: LatticeDesc,
    acceptance_desc: AcceptanceDesc,
    burn_in_desc: BurnInDesc,
    snapshot: Option<SnapshotDesc>,
}

impl SystemBuilder {
    /// Creates a new system builder with default configuration.
    pub fn new() -> Self {
        Self {
            snapshot: None,
            lattice_desc: LatticeDesc::default(),
            param_desc: ParamDesc::default(),
            acceptance_desc: AcceptanceDesc::default(),
            burn_in_desc: BurnInDesc::default(),
        }
    }

    /// Enables snapshots using the given configuration.
    ///
    /// Snapshots are disabled by default.
    ///
    /// See [`SnapshotDesc`] for more information.
    pub fn enable_snapshot(mut self, desc: SnapshotDesc) -> Self {
        self.snapshot = Some(desc);
        self
    }

    /// Disables snapshots.
    ///
    /// Snapshots are disabled by default.
    pub fn disable_snapshot(mut self) -> Self {
        self.snapshot = None;
        self
    }

    /// Sets the configuration for burn in.
    ///
    /// See [`BurnInDesc`] for more information.
    pub fn with_burn_in(mut self, desc: BurnInDesc) -> Self {
        self.burn_in_desc = desc;
        self
    }

    /// Sets the configuration for the acceptance ratio.
    ///
    /// See [`AcceptanceDesc`] for more information.
    pub fn with_acceptance(mut self, desc: AcceptanceDesc) -> Self {
        self.acceptance_desc = desc;
        self
    }

    /// Sets the lattice configuration.
    ///
    /// See [`LatticeDesc`] for more information.
    pub fn with_lattice(mut self, desc: LatticeDesc) -> Self {
        self.lattice_desc = desc;
        self
    }

    /// Sets the physical constants.
    ///
    /// See [`ParamDesc`] for more information.
    pub fn with_params(mut self, desc: ParamDesc) -> Self {
        self.param_desc = desc;
        self
    }

    /// Creates the simulation using the given options.
    pub fn build(self) -> anyhow::Result<System> {
        tracing::info!("Generating simulation with configuration: {self:?}");

        let lattice = ScalarLattice4D::new(self.lattice_desc);

        let snapshot_state = self
            .snapshot
            .map(|desc| {
                let snapshot = hdf5::File::append(&desc.file)?;

                let elapsed = UNIX_EPOCH.elapsed()?.as_secs();
                let [st, sx, sy, sz] = lattice.dimensions();

                let [ct, cx, cy, cz] = desc.chunk_size;

                let set_name = format!("data-{elapsed}");
                let set = snapshot
                    .new_dataset::<f64>()
                    .chunk([1, ct, cx, cy, cz])
                    .shape((1.., st, sx, sy, sz))
                    .shuffle()
                    .fletcher32()
                    .deflate(5)
                    .create(set_name.as_str())?;

                tracing::debug!("Dataset {set_name} created");

                Ok::<SnapshotState, anyhow::Error>(SnapshotState {
                    file: Some(snapshot),
                    dataset: Arc::new(set),
                    desc,
                    tx: None,
                    thread: None,
                })
            })
            .transpose()?;

        let mut sim = System {
            simulating: AtomicBool::new(false),
            current_sweep: 0,
            lattice,
            mass_squared: self.param_desc.mass_squared,
            coupling: self.param_desc.coupling,
            stats: SystemStats::default(),
            current_step_size: AtomicF64::new(self.acceptance_desc.initial_step_size),
            acceptance_desc: self.acceptance_desc,
            burn_in_desc: self.burn_in_desc,
            correlation_slices: Vec::new(),
            measurement_interval: 50,
            snapshot_state,
        };

        let first_action = sim.compute_full_action();
        sim.stats
            .current_action
            .store(first_action, Ordering::Release);

        sim.stats.action_history.push(first_action);

        tracing::info!("System initialised");

        Ok(sim)
    }
}

impl Default for SystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}
