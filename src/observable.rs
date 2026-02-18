//! Implements the observable API.

use rayon::prelude::*;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault};
use nohash_hasher::NoHashHasher;
use crate::sim::System;

pub struct ObservableStorage {
    map: HashMap<TypeId, Box<dyn BaseObservableState>, BuildHasherDefault<NoHashHasher<u64>>>,
}

impl ObservableStorage {
    /// Creates a new observable storage.
    pub fn new() -> Self {
        Self {
            map: HashMap::with_hasher(BuildHasherDefault::<NoHashHasher<u64>>::default()),
        }
    }

    pub fn measure(&mut self, system: &mut System) {
        self.map.par_iter().for_each(|(k, v)| {

        })
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
pub enum MeasureInterval {
    /// Every `n` autocorrelation times.
    Autocorrelation(u64),
    /// Every `n` sweeps.
    Sweep(u64)
}

/// General functionality that is independent of measurement type.
pub trait BaseObservableState: Send + Sync + 'static {
    /// Clears the observable's state.
    fn clear(&mut self);
    /// Reserves capacity for `n` additional measurements.
    fn reserve(&mut self, n: usize);
}

/// Information specific to certain measurement types.
pub trait ObservableState<O: Copy = f64>: BaseObservableState {
    fn measure(&mut self, system: &System);
    /// Returns the last measured quantity.
    fn measured(&self) -> Option<O>;
}

pub trait Observable: 'static {
    type Output: Copy;
    type State: ObservableState<Self::Output>;

    const NAME: &'static str;

    fn interval() -> MeasureInterval;
    fn measure(system: &System) -> Self::Output;
}