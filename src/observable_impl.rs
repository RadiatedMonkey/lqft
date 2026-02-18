//! Implementations of basic observables.

use crate::observable::*;

use crate::observable::{MeasureInterval, Observable, ObservableState};
use crate::sim::System;

/// Measures the mean of the lattice.
pub struct MeanValue;

impl Observable for MeanValue {
    type Output = f64;
    type State = MeanValueState;

    const NAME: &'static str = "mean_value";

    #[inline]
    fn interval() -> MeasureInterval {
        MeasureInterval::Sweep(1)
    }

    #[inline]
    fn measure(system: &System) -> f64 {
        system.lattice().mean()
    }
}

pub struct MeanValueState {
    data: Vec<f64>
}

impl BaseObservableState for MeanValueState {
    fn clear(&mut self) {
        self.data.clear();
    }

    fn reserve(&mut self, n: usize) {
        self.data.reserve(n);
    }
}

impl ObservableState for MeanValueState {
    fn measure(&mut self, system: &System) {
        let mean = system.lattice().mean();
        self.data.push(mean);
    }

    fn measured(&self) -> Option<f64> {
        self.data.last().copied()
    }
}

/// Measures the variance of the lattice.
pub struct Variance;

impl Observable for Variance {
    type Output = f64;
    type State = VarianceState;

    const NAME: &'static str = "variance";

    #[inline]
    fn interval() -> MeasureInterval {
        MeasureInterval::Sweep(1)
    }

    #[inline]
    fn measure(system: &System) -> f64 {
        todo!()
    }
}

pub struct VarianceState {
    data: Vec<f64>
}

impl BaseObservableState for VarianceState {
    fn clear(&mut self) {
        self.data.clear();
    }

    fn reserve(&mut self, n: usize) {
        self.data.reserve(n);
    }
}

impl ObservableState<f64> for VarianceState {
    fn measure(&mut self, system: &System) {
        let var = system.lattice().variance();
        self.data.push(var);
    }

    fn measured(&self) -> Option<f64> {
        self.data.last().copied()
    }
}