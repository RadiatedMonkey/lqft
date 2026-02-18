//! Implementations of basic observables.

use crate::observable::*;

use crate::observable::{MeasureFrequency, Observable, ObservableState};
use crate::sim::{System, SystemData};

/// Measures the mean of the lattice.
pub struct MeanValue;

impl Observable for MeanValue {
    type Output = f64;
    type State = MeanValueState;

    const NAME: &'static str = "mean_value";
}

pub struct MeanValueState {
    data: Vec<f64>
}

impl ObservableMeasure for MeanValueState {
    fn measure(&mut self, data: &SystemData) {
        let mean = data.lattice.mean();
        self.data.push(mean);
    }
}

impl ObservableState for MeanValueState {
    const FREQUENCY: MeasureFrequency = MeasureFrequency::Sweep(1);

    fn measured(&self) -> Option<f64> {
        self.data.last().copied()
    }

    fn init() -> Self {
        Self { data: Vec::new() }
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
    type Output = f64;
    type State = VarianceState;

    const NAME: &'static str = "variance";
}

pub struct VarianceState {
    data: Vec<f64>
}

impl ObservableMeasure for VarianceState {
    fn measure(&mut self, data: &SystemData) {
        let var = data.lattice.variance();
        self.data.push(var);
    }
}

impl ObservableState<f64> for VarianceState {
    const FREQUENCY: MeasureFrequency = MeasureFrequency::Sweep(1);

    fn measured(&self) -> Option<f64> {
        self.data.last().copied()
    }

    fn init() -> Self {
        Self { data: Vec::new() }
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    fn prepare(&mut self, n: usize) {
        self.data.reserve(n);
    }
}