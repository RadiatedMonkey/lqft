use std::sync::atomic::{AtomicUsize, Ordering};
use atomic_float::AtomicF64;
use crate::setup::SnapshotType;
use crate::sim::System;
use crate::snapshot::SnapshotFragment;

/// Statistics of the simulation. Every finished sweep, a new statistic is recorded.
pub struct SystemStats {
    /// Total moves made in the simulation
    pub total_moves: AtomicUsize,
    /// Total moves accepted
    pub accepted_moves: AtomicUsize,
    /// History of accepted moves. A new datapoint is recorded on every statistics interval.
    pub accepted_move_history: Vec<usize>,
    /// History of accepted move ratio.
    pub accept_ratio_history: Vec<f64>,
    /// History of the step size.
    pub step_size_history: Vec<f64>,
    /// History of the field mean
    pub mean_history: Vec<f64>,
    /// History of the field variance.
    pub meansq_history: Vec<f64>,
    /// History of the action over time.
    pub action_history: Vec<f64>,
    /// The action at the current point in time.
    pub current_action: AtomicF64,
    /// The history of the thermalisation ratio.
    pub thermalisation_ratio_history: Vec<f64>,
    /// The sweep at which the system first passed the thermalisation threshold.
    pub thermalised_at: Option<usize>,
    /// The amount of measurements performed after thermalisation.
    pub performed_measurements: usize,
    pub snapshot_batch: Vec<SnapshotFragment>
}

/// The stats of the current sweep.
pub struct SweepStats {
    pub total_moves: usize,
    pub accepted_moves: usize,
    pub accept_ratio: f64,
    pub step_size: f64,
    pub mean: f64,
    pub meansq: f64,
    pub action: f64,
    pub th_ratio: f64,
    pub performed_measurements: usize
}

impl System {
    /// Records statistics on the current sweep.
    pub(crate) fn record_stats(&mut self) -> anyhow::Result<()> {
        let mean = self.lattice.mean();
        let meansq = self.lattice.meansq();
        let action = self.stats.current_action.load(Ordering::Acquire);

        self.stats.mean_history.push(mean);
        self.stats.meansq_history.push(meansq);
        self.stats.action_history.push(action);

        let accept = self.stats.accepted_moves();
        let total = self.stats.total_moves();
        let ratio = accept as f64 / total as f64 * 100.0;

        self.stats.accepted_move_history.push(accept);
        self.stats.accept_ratio_history.push(ratio);
        self.stats.step_size_history.push(self.current_step_size());

        if let Some(snapshot) = &self.snapshot_state {
            if let SnapshotType::Interval(interval) = snapshot.desc.ty {
                // Check whether a snapshot should be saved
                if total % interval == 0 {
                    let sweep = SweepStats {
                        total_moves: total,
                        accepted_moves: accept,
                        accept_ratio: ratio,
                        step_size: self.current_step_size(),
                        mean,
                        meansq: 0.0,
                        action,
                        th_ratio: 0.0,
                        performed_measurements: 0,
                    };

                    let fragment = SnapshotFragment {
                        lattice: unsafe { self.lattice.clone() },
                        stats: sweep
                    };

                    snapshot.send_fragment(fragment)?;
                }
            }
        }

        Ok(())
    }

    pub fn finalize_stats(&mut self) -> anyhow::Result<()> {
        todo!()
    }
}

impl SystemStats {
    /// Reserves enough space for `count` additional statistics.
    ///
    /// This is called at the start of the simulation with the desired sweep count to improve performance.
    pub fn reserve_capacity(&mut self, count: usize) {
        self.accepted_move_history.reserve(count);
        self.accept_ratio_history.reserve(count);
        self.step_size_history.reserve(count);
        self.mean_history.reserve(count);
        self.meansq_history.reserve(count);
        self.action_history.reserve(count);
        self.thermalisation_ratio_history.reserve(count);
    }

    /// The most recent value of the whole system action.
    pub fn current_action(&self) -> f64 {
        self.current_action.load(Ordering::Relaxed)
    }

    /// The total amount of attempted moves so far.
    pub fn total_moves(&self) -> usize {
        self.total_moves.load(Ordering::Relaxed)
    }

    /// The total amount of accepted moves so far.
    pub fn accepted_moves(&self) -> usize {
        self.accepted_moves.load(Ordering::Relaxed)
    }
}

impl Default for SystemStats {
    fn default() -> Self {
        Self {
            current_action: AtomicF64::new(0.0),
            total_moves: AtomicUsize::new(0),
            accepted_move_history: Vec::new(),
            accepted_moves: AtomicUsize::new(0),
            accept_ratio_history: Vec::new(),
            step_size_history: Vec::new(),
            mean_history: Vec::new(),
            meansq_history: Vec::new(),
            action_history: Vec::new(),
            thermalisation_ratio_history: Vec::new(),
            thermalised_at: None,
            performed_measurements: 0,
            snapshot_batch: Vec::new()
        }
    }
}