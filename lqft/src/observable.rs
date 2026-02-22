//! Implements the observable API.

use rayon::prelude::*;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault};
use nohash_hasher::NoHashHasher;
use crate::sim::{System, SystemData};

pub struct ObservableRegistry<const Dim: usize> {
    map: HashMap<TypeId, Box<dyn ObservableState<Dim>>, BuildHasherDefault<NoHashHasher<u64>>>,
}

impl<const Dim: usize> ObservableRegistry<Dim> {
    /// Creates a new observable storage.
    pub fn new() -> Self {
        Self {
            map: HashMap::with_hasher(BuildHasherDefault::<NoHashHasher<u64>>::default()),
        }
    }

    pub fn measure(&mut self, data: &SystemData<Dim>) {
        self.map.par_iter_mut().for_each(|(_k, v)| {
            if v.should_measure(data) {
                v.measure(data);
            }
        })
    }

    pub fn register<O: Observable<Dim>>(&mut self) {
        self.map.insert(TypeId::of::<O>(), Box::new(O::new_state()));
    }

    /// Retrieves the state of the given observable.
    pub fn get<O: Observable<Dim>>(&self) -> Option<&O::State> {
        self.map.get(&TypeId::of::<O>()).map(|boxed| {
            let any = boxed.as_any();
            boxed.as_any().downcast_ref::<O::State>()
        }).flatten()
    }

    /// Mutably retrieves the state of the given observable.
    pub fn get_mut<O: Observable<Dim>>(&mut self) -> Option<&mut O::State> {
        self.map.get_mut(&TypeId::of::<O>()).map(|boxed| {
            // boxed.as_any_mut().downcast_mut::<O::State>()
            todo!()
        }).flatten()
    }

    /// Retrieves the last measured value of the observable.
    pub fn measured<O: Observable<Dim>>(&self) -> Option<f64> {
        let obs = self.get::<O>()?;
        obs.measured()
    }
}

/// When to perform measurements.
pub enum MeasureFrequency {
    Once,
    /// Every `n` autocorrelation times.
    Autocorrelation(usize),
    /// Every `n` sweeps.
    Sweep(usize)
}

// trait AsAny: Any {
//     fn as_any(&self) -> &dyn Any;
//     fn as_any_mut(&mut self) -> &mut dyn Any;
// }
//
// impl<T: Any> AsAny for T {
//     fn as_any(&self) -> &dyn Any { self }
//     fn as_any_mut(&mut self) -> &mut dyn Any { self }
// }

/// Information specific to certain measurement types.
pub trait ObservableState<const Dim: usize>: Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;

    /// Approximate frequency of measurements.
    fn frequency(&self) -> MeasureFrequency;
    /// Whether this observable requires thermalisation.
    fn burn_in(&self) -> bool { true }

    /// Makes a new measurement.
    fn measure(&mut self, data: &SystemData<Dim>);

    /// Determines whether the observables should be measured.
    ///
    /// By default this follows the frequency defined in [`FREQUENCY`](Self::FREQUENCY).
    fn should_measure(&self, data: &SystemData<Dim>) -> bool {
        if self.burn_in() && !data.stats.thermalised_at.is_some() {
            return false
        }

        match self.frequency() {
            MeasureFrequency::Autocorrelation(_) => todo!(),
            MeasureFrequency::Sweep(n) => data.stats.current_sweep % n == 0,
            MeasureFrequency::Once => panic!("One-time observables should have a custom `should_measure` implementation")
        }
    }

    /// Returns the last measured quantity.
    fn measured(&self) -> Option<f64>;

    /// Clears the observable's state.
    fn clear(&mut self);

    /// Prepares for `n` additional measurements. The number of measurements is an estimate based
    /// on [`FREQUENCY`](Self::FREQUENCY).
    fn prepare(&mut self, n: usize);
}

pub trait Observable<const Dim: usize>: 'static {
    type State: ObservableState<Dim>;

    /// Name of the observable.
    const NAME: &'static str;

    fn new_state() -> Self::State;
}