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
    pub fn compute_burn_in_ratio(&self) -> (f64, bool) {
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

#[derive(Debug, Clone)]
pub struct LatticeDesc {
    pub initial_state: InitialState,
    pub dimensions: [usize; 4],
    pub spacing: f64
}

impl Default for LatticeDesc {
    fn default() -> Self {
        Self {
            initial_state: InitialState::RandomRange(-0.5..0.5),
            dimensions: [40, 20, 20, 20],
            spacing: 1.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct AcceptanceDesc {
    pub desired_range: Range<f64>,
    pub correction_size: f64,
    pub correction_interval: usize,
    pub initial_step_size: f64
}

impl Default for AcceptanceDesc {
    fn default() -> Self {
        Self {
            desired_range: 0.3..0.5,
            correction_size: 0.05,
            correction_interval: 20_000,
            initial_step_size: 1.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct BurnInDesc {
    pub avg_block_size: usize,
    pub desired_ratio: f64
}

impl Default for BurnInDesc {
    fn default() -> Self {
        Self {
            avg_block_size: 100,
            desired_ratio: 0.1
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
    Checkpoint
}

pub struct SnapshotDesc {
    pub file: String,
    pub ty: SnapshotType,
    pub chunk_size: [usize; 4],
    pub flush_method: FlushMethod
}

impl Default for SnapshotDesc {
    fn default() -> Self {
        Self {
            file: "snapshots/snapshots.h5".to_owned(),
            ty: SnapshotType::Checkpoint,
            chunk_size: [16; 4],
            flush_method: FlushMethod::Sequential
        }
    }
}

/// Used to configure a lattice simulation.
pub struct SystemBuilder {
    lattice_desc: LatticeDesc,
    acceptance_desc: AcceptanceDesc,
    burn_in_desc: BurnInDesc,
    snapshot: Option<SnapshotDesc>,

    mass_squared: f64,
    bare_coupling: f64,
}

impl SystemBuilder {
    pub fn new() -> Self {
        Self {
            snapshot: None,
            lattice_desc: LatticeDesc::default(),
            mass_squared: 1.0,
            bare_coupling: 0.0,
            acceptance_desc: AcceptanceDesc::default(),
            burn_in_desc: BurnInDesc::default()
        }
    }

    pub fn enable_snapshot(mut self, desc: SnapshotDesc) -> Self {
        self.snapshot = Some(desc);
        self
    }

    pub fn disable_snapshot(mut self) -> Self {
        self.snapshot = None;
        self
    }

    pub fn with_burn_in(mut self, desc: BurnInDesc) -> Self {
        self.burn_in_desc = desc;
        self
    }

    pub fn with_acceptance(mut self, desc: AcceptanceDesc) -> Self {
        self.acceptance_desc = desc;
        self
    }

    pub fn with_lattice(mut self, desc: LatticeDesc) -> Self {
        self.lattice_desc = desc;
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

    /// Creates the simulation using the given options.
    pub fn build(self) -> anyhow::Result<System> {
        let lattice = ScalarLattice4D::new(self.lattice_desc);

        let snapshot_state = self.snapshot.map(|desc| {
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

            println!("Dataset {set_name} created");

            Ok::<SnapshotState, anyhow::Error>(SnapshotState {
                file: snapshot,
                dataset: Arc::new(set),
                desc,
                tx: None,
                thread: None
            })
        }).transpose()?;

        let mut sim = System {
            simulating: AtomicBool::new(false),
            current_sweep: 0,
            lattice,
            mass_squared: self.mass_squared,
            coupling: self.bare_coupling,
            stats: SystemStats::default(),
            current_step_size: AtomicF64::new(self.acceptance_desc.initial_step_size),
            acceptance_desc: self.acceptance_desc,
            burn_in_desc: self.burn_in_desc,
            correlation_slices: Vec::new(),
            measurement_interval: 50,
            snapshot_state
        };

        let first_action = sim.compute_full_action();
        sim.stats
            .current_action
            .store(first_action, Ordering::Release);

        sim.stats.action_history.push(first_action);

        // Reserves capacity for batches in case flush method uses batching.
        if let Some(SnapshotState { desc, .. }) = &sim.snapshot_state {
            let SnapshotDesc {
                flush_method, ..
            } = desc;

            if let FlushMethod::Batched(size) = flush_method {
                sim.stats.snapshot_batch.reserve(*size);
            }
        }

        Ok(sim)
    }
}

impl Default for SystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}