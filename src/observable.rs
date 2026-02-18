//! Implements the observable API.

use rayon::prelude::*;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault};
use nohash_hasher::NoHashHasher;
use crate::sim::{System, SystemData};

pub struct ObservableRegistry {
    map: HashMap<TypeId, Box<dyn ObservableMeasure>, BuildHasherDefault<NoHashHasher<u64>>>,
}

impl ObservableRegistry {
    /// Creates a new observable storage.
    pub fn new() -> Self {
        Self {
            map: HashMap::with_hasher(BuildHasherDefault::<NoHashHasher<u64>>::default()),
        }
    }

    pub fn measure(&mut self, data: &SystemData) {
        self.map.par_iter_mut().for_each(|(k, v)| {
            v.measure(data)
        })
    }

    pub fn register<O: Observable>(&mut self) {
        self.map.insert(TypeId::of::<O::State>(), Box::new(O::State::init()));
    }

    /// Retrieves the state of the given observable.
    pub fn get<O: Observable>(&self) -> Option<&O::State> {
        self.map.get(&TypeId::of::<O>()).map(|boxed| {
            let any = boxed as &dyn Any;
            any.downcast_ref::<O::State>()
        }).flatten()
    }

    /// Mutably retrieves the state of the given observable.
    pub fn get_mut<O: Observable>(&mut self) -> Option<&mut O::State> {
        self.map.get_mut(&TypeId::of::<O>()).map(|boxed| {
            let any = boxed as &mut dyn Any;
            any.downcast_mut::<O::State>()
        }).flatten()
    }

    /// Retrieves the last measured value of the observable.
    pub fn measured<O: Observable>(&self) -> Option<O::Output> {
        let obs = self.get::<O>()?;
        obs.measured()
    }
}

/// When to perform measurements.
pub enum MeasureFrequency {
    /// Every `n` autocorrelation times.
    Autocorrelation(usize),
    /// Every `n` sweeps.
    Sweep(usize),

}

/// General functionality that is independent of measurement type.
pub trait ObservableMeasure: Send + Sync + 'static {
    /// Makes a new measurement.
    fn measure(&mut self, data: &SystemData);
}

/// Information specific to certain measurement types.
pub trait ObservableState<O: Copy = f64>: ObservableMeasure {
    /// Approximate frequency of measurements.
    const FREQUENCY: MeasureFrequency;

    /// Determines whether the observables should be measured.
    ///
    /// By default this follows the frequency defined in [`FREQUENCY`](Self::FREQUENCY).
    fn should_measure(&self, data: &SystemData) -> bool {
        match Self::FREQUENCY {
            MeasureFrequency::Autocorrelation(_) => todo!(),
            MeasureFrequency::Sweep(n) => data.stats.current_sweep % n == 0
        }
    }

    /// Returns the last measured quantity.
    fn measured(&self) -> Option<O>;

    /// Creates a new state.
    fn init() -> Self;

    /// Clears the observable's state.
    fn clear(&mut self);

    /// Prepares for `n` additional measurements. The number of measurements is an estimate based
    /// on [`FREQUENCY`](Self::FREQUENCY).
    fn prepare(&mut self, n: usize);
}

pub trait Observable: 'static {
    type Output: Copy;
    type State: ObservableState<Self::Output>;

    /// Name of the observable.
    const NAME: &'static str;
}