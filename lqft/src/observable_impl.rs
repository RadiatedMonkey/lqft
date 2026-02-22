//! Implementations of basic observables.

use std::any::Any;

use crate::observable::{MeasureFrequency, Observable, ObservableState};
use crate::sim::SystemData;
use crate::util::FType;

/// Measures the mean of the lattice.
pub struct MeanValue;

impl<const Dim: usize> Observable<Dim> for MeanValue {
    type State = MeanValueState;

    const NAME: &'static str = "mean_value";

    fn new_state() -> Self::State {
        MeanValueState { data: Vec::new() }
    }
}

pub struct MeanValueState {
    data: Vec<FType>
}

impl<const Dim: usize> ObservableState<Dim> for MeanValueState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn frequency(&self) -> MeasureFrequency {
        MeasureFrequency::Sweep(1)
    }

    fn burn_in(&self) -> bool {
        false
    }

    fn measure(&mut self, data: &SystemData<Dim>) {
        let mean = data.lattice.mean_seq();
        self.data.push(mean);
    }

    fn measured(&self) -> Option<FType> {
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

impl<const Dim: usize> Observable<Dim> for Variance {
    type State = VarianceState;

    const NAME: &'static str = "variance";

    fn new_state() -> Self::State {
        VarianceState { data: Vec::new() }
    }
}

pub struct VarianceState {
    data: Vec<FType>
}

impl<const Dim: usize> ObservableState<Dim> for VarianceState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn frequency(&self) -> MeasureFrequency {
        MeasureFrequency::Sweep(1)
    }

    fn burn_in(&self) -> bool {
        false
    }

    fn measure(&mut self, data: &SystemData<Dim>) {
        let mean = data.lattice.variance();
        self.data.push(mean);
    }

    fn measured(&self) -> Option<FType> {
        self.data.last().copied()
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    fn prepare(&mut self, n: usize) {
        self.data.reserve(n);
    }
}