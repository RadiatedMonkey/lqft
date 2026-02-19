//! Implementations of basic observables.

use std::any::Any;
use crate::observable::*;

use crate::observable::{MeasureFrequency, Observable, ObservableState};
use crate::sim::{System, SystemData};

/// Measures the mean of the lattice.
pub struct MeanValue;

impl Observable for MeanValue {
    type State = MeanValueState;

    const NAME: &'static str = "mean_value";

    fn new_state() -> Self::State {
        MeanValueState { data: Vec::new() }
    }
}

pub struct MeanValueState {
    data: Vec<f64>
}

impl ObservableState for MeanValueState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn frequency(&self) -> MeasureFrequency {
        MeasureFrequency::Sweep(1)
    }

    fn burn_in(&self) -> bool {
        false
    }

    fn measure(&mut self, data: &SystemData) {
        let mean = data.lattice.mean();
        self.data.push(mean);
    }

    fn measured(&self) -> Option<f64> {
        self.data.last().copied()
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    fn prepare(&mut self, n: usize) {
        self.data.reserve(n);
    }
}

/// Measures the variance of the lattice.
pub struct Variance;

impl Observable for Variance {
    type State = VarianceState;

    const NAME: &'static str = "variance";

    fn new_state() -> Self::State {
        VarianceState { data: Vec::new() }
    }
}

pub struct VarianceState {
    data: Vec<f64>
}

impl ObservableState for VarianceState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn frequency(&self) -> MeasureFrequency {
        MeasureFrequency::Sweep(1)
    }

    fn burn_in(&self) -> bool {
        false
    }

    fn measure(&mut self, data: &SystemData) {
        let mean = data.lattice.variance();
        self.data.push(mean);
    }

    fn measured(&self) -> Option<f64> {
        self.data.last().copied()
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    fn prepare(&mut self, n: usize) {
        self.data.reserve(n);
    }
}