//! Implementations of basic observables.

use crate::observable::{Observable};
use crate::sim::SystemData;

/// Measures the mean of the lattice.
pub struct MeanValue {
    data: Vec<f64>
}

impl Observable for MeanValue {
    type Output = f64;

    const NAME: &'static str = "mean";

    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn reserve(&mut self, n: usize) {
        self.data.reserve(n);
    }

    fn observe(&mut self, system: &SystemData) {
        self.data.push(system.lattice.mean())
    }

    fn latest(&self) -> Option<f64> {
        self.data.last().copied()
    }
}

/// Measures the variance of the lattice.
pub struct Variance {
    data: Vec<f64>
}

impl Observable for Variance {
    type Output = f64;

    const NAME: &'static str = "variance";

    fn new() -> Variance {
        Variance { data: Vec::new() }
    }

    fn reserve(&mut self, n: usize) {
        self.data.reserve(n);
    }

    fn observe(&mut self, system: &SystemData) {
        self.data.push(system.lattice.variance());
    }

    fn latest(&self) -> Option<f64> {
        self.data.last().copied()
    }
}