//! Functionality related to initialisation of the system

use std::sync::atomic::Ordering;

use crate::sim::System;

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