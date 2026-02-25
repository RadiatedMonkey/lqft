//! Implementations of basic observables.

use crate::observable::{Observable};
use crate::sim::SystemData;

/// Measures the mean of the lattice.
#[derive(Debug)]
pub struct Mean {
    history: Vec<f64>
}

impl Observable for Mean {
    type Output = f64;

    const NAME: &'static str = "mean";

    fn new() -> Self {
        Self { history: Vec::new() }
    }

    fn reserve(&mut self, n: usize) {
        self.history.reserve(n);
    }

    fn observe(&mut self, system: &SystemData) {
        self.history.push(system.lattice.mean())
    }

    fn history(&self) -> &[f64] {
        &self.history
    }

    fn latest(&self) -> Option<f64> {
        self.history.last().copied()
    }
}

/// Measures the variance of the lattice.
#[derive(Debug)]
pub struct Variance {
    history: Vec<f64>
}

impl Observable for Variance {
    type Output = f64;

    const NAME: &'static str = "variance";

    fn new() -> Variance {
        Variance { history: Vec::new() }
    }

    fn reserve(&mut self, n: usize) {
        self.history.reserve(n);
    }

    fn observe(&mut self, system: &SystemData) {
        self.history.push(system.lattice.variance());
    }

    fn history(&self) -> &[f64] {
        &self.history
    }

    fn latest(&self) -> Option<f64> {
        self.history.last().copied()
    }
}

#[derive(Debug)]
pub struct ActionDensity {
    history: Vec<f64>
}

impl Observable for ActionDensity {
    type Output = f64;

    const NAME: &'static str = "action_density";

    fn new() -> ActionDensity {
        ActionDensity {
            history: Vec::new()
        }
    }

    fn reserve(&mut self, n: usize) {
        self.history.reserve(n);
    }

    fn observe(&mut self, system: &SystemData) {
        self.history.push(system.compute_full_action());
    }

    fn history(&self) -> &[f64] {
        &self.history
    }

    fn latest(&self) -> Option<f64> {
        self.history.last().copied()
    }
}