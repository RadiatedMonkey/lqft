//! Implements the observable API.

use std::any::{Any, TypeId};
use std::fmt::Debug;
use std::marker::PhantomData;
use realfft::RealFftPlanner;
use crate::observable_impl::Mean;
use crate::sim::SystemData;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CorrelationResult {
    /// The autocorrelation time has been computed successfully.
    Success(f64), 
    Sampling,
    Disabled
}

pub trait CorrelationList {
    fn is_success(&self) -> bool;
}

impl CorrelationList for () {
    fn is_success(&self) -> bool {
        false
    }
}

impl CorrelationList for CorrelationResult {
    fn is_success(&self) -> bool {
        match self { Self::Success(_) => true, _ => false }
    }
}

impl<U: CorrelationList> CorrelationList for (U, CorrelationResult) {
    fn is_success(&self) -> bool {
        self.0.is_success() && self.1.is_success()
    }
}

pub struct ObservableMeta {
    autocor_time: f64
}

impl ObservableMeta {
    pub fn new() -> ObservableMeta {
        ObservableMeta {
            autocor_time: 0.0
        }
    }
}

/// A list of different observables.
pub trait ObservableHList: 'static {
    /// The new type when `O` is adjoined to the list.
    type Append<O: Observable>: ObservableHList;

    /// The amount of observables in this list.
    const LEN: usize;

    /// Creates a new empty list.
    fn new() -> Self;

    /// Determines whether an observable is in the list.
    fn has<O: Observable>() -> bool;

    /// Collects data for all observables.
    fn observe(&mut self, system: &SystemData);

    // Returns the computed autocorrelation time of the observable.
    fn autocorrelation<O: Observable>(&self) -> Option<f64>;

    /// Returns the measurement history for the given observable.
    ///
    /// This function panics if the given observable is not in the list.
    fn history<O: Observable>(&self) -> &[O::Output];

    /// Returns the last measured result for the given observable.
    /// 
    /// This function returns `None` if the observable has produced no measurements yet, or
    /// the observable is not in the list.
    fn measured<O: Observable>(&self) -> Option<O::Output>;

    /// Reserves capacity for an additional `n` measurements.
    fn reserve(&mut self, n: usize);

    /// Retrieves the state.
    fn state<O: Observable>(&self) -> &O;

    /// Retrieves the state mutably.
    fn state_mut<O: Observable>(&mut self) -> &mut O;
}

impl ObservableHList for () {
    type Append<O: Observable> = (ObservableMeta, O);

    const LEN: usize = 0;

    fn new() -> () {}

    fn has<O: Observable>() -> bool { 
        false 
    }

    fn observe(&mut self, _system: &SystemData) {}

    fn autocorrelation<O: Observable>(&self) -> Option<f64> { None }

    fn history<O: Observable>(&self) -> &[O::Output] {
        panic!("Cannot retrieve history. Observable \"{}\" is not registered. Register it using `SystemBuilder::with_observable`", O::NAME);
    }

    fn measured<O: Observable>(&self) -> Option<O::Output> {
        panic!("Cannot retrieve last measurement. Observable \"{}\" is not registered. Register it using `SystemBuilder::with_observable.`", O::NAME);
    }

    fn reserve(&mut self, _n: usize) {}

    fn state<O: Observable>(&self) -> &O {
        panic!("Cannot retrieve observable state. Observable \"{}\" is not registered. Register it using `SystemBuilder::with_observable`.", O::NAME);
    }

    fn state_mut<O: Observable>(&mut self) -> &mut O {
        panic!("Cannot retrieve observable state. Observable \"{}\" is not registered. Register it using `SystemBuilder::with_observable`.", O::NAME);
    }
}

impl<O1: Observable> ObservableHList for (ObservableMeta, O1) {
    type Append<O: Observable> = ((ObservableMeta, O1), (ObservableMeta, O));

    const LEN: usize = 1;

    fn new() -> (ObservableMeta, O1) { (ObservableMeta::new(), <O1 as Observable>::new()) }

    fn has<O: Observable>() -> bool { 
        TypeId::of::<O>() == TypeId::of::<O1>() 
    }

    fn observe(&mut self, system: &SystemData) {
        <O1 as Observable>::observe(&mut self.1, system);
    }

    fn autocorrelation<O: Observable>(&self) -> Option<f64> {
        todo!()
    }

    // fn autocorrelation<O: Observable>(&self) -> Option<f64> where O::Output: Into<f64> {
    //     let cast = (self as &dyn Any)
    //         .downcast_ref::<O>()
    //         .expect("Autocorrelation cast failed, this is a bug.");

    //     let series = cast.history();

    //     let series_cast = series.iter().map(|&f| <O::Output as Into<f64>>::into(f));
    //     let series_mean = series_cast.clone().sum::<f64>() / series.len() as f64;
        
    //     let mut series_2n = Vec::with_capacity(2 * series.len());
    //     series_2n.extend(series_cast.map(|f| f - series_mean));
    //     series_2n.resize(2 * series.len(), 0.0);

    //     let mut planner = RealFftPlanner::<f64>::new();
    //     let fft = planner.plan_fft_forward(series_2n.len());

    //     let mut freq = fft.make_output_vec();
    //     fft.process(&mut series_2n, &mut freq);

    //     freq.iter_mut().for_each(|s| *s = *s * s.conj());

    //     let inv_fft = planner.plan_fft_inverse(series_2n.len());
    //     let mut autocov = inv_fft.make_output_vec();
    //     inv_fft.process(&mut freq, &mut autocov);

    //     let c0 = autocov[0];
    //     let rho = autocov[..series.len()].iter().map(|&c| c / c0);
    //     let tau_int = 0.5 + rho.skip(1).take_while(|&r| r > 0.0).sum::<f64>();

    //     println!("Autocorrelation time: {tau_int}");

    //     Some(tau_int)

    //     // if self.data.autocorrelation_samples.len() == AUTOCOR_SAMPLE_SIZE {
    //         //     // Estimate autocorrelation time
    //         //     let series_mean = self.data.autocorrelation_samples.iter().sum::<f64>() / AUTOCOR_SAMPLE_SIZE as f64;
    //         //     let mut series = self.data.autocorrelation_samples.iter().map(|m| m - series_mean).collect::<Vec<_>>();
    //         //     series.resize(2 * series.len(), 0.0);

    //         //     let mut planner = RealFftPlanner::<f64>::new();
    //         //     let fft = planner.plan_fft_forward(series.len());

    //         //     let mut freq = fft.make_output_vec();
    //         //     fft.process(&mut series, &mut freq).unwrap();

    //         //     freq.iter_mut().for_each(|s| *s = *s * s.conj());

    //         //     let inv_fft = planner.plan_fft_inverse(series.len());
    //         //     let mut autocov = inv_fft.make_output_vec();
    //         //     inv_fft.process(&mut freq, &mut autocov).unwrap();

    //         //     let c0 = autocov[0];
    //         //     let rho: Vec<f64> = autocov[..self.data.autocorrelation_samples.len()].iter().map(|&c| c / c0).collect();

    //         //     let tau_int = 0.5 + rho[1..].iter().take_while(|&r| *r > 0.0).sum::<f64>();
    //         //     println!("Autocorrelation time: {tau_int}");
    //         // }
    // }

    fn history<O: Observable>(&self) -> &[O::Output] {
        if !Self::has::<O>() {
            unreachable!("This path should not be reached. This is a bug");
        } else {
            let cast = (self as &dyn Any)
                .downcast_ref::<O>()
                .expect("O != O1, this is a bug. In O1");
            <O as Observable>::history(cast)
        }
    }

    fn measured<O: Observable>(&self) -> Option<O::Output> {
        if !Self::has::<O>() {
            unreachable!("This path should not be reached. This is a bug");
        } else {
            let cast = (self as &dyn Any)
                .downcast_ref::<O>()
                .expect("O != O1, this is a bug. In O1");
            <O as Observable>::latest(cast)
        }
    }

    fn reserve(&mut self, n: usize) {
        <O1 as Observable>::reserve(&mut self.1, n);
    }

    fn state<O: Observable>(&self) -> &O {
        (self as &dyn Any).downcast_ref::<O>().expect("State cast failed. This is a bug.")
    }

    fn state_mut<O: Observable>(&mut self) -> &mut O {
        (self as &mut dyn Any).downcast_mut::<O>().expect("State cast failed. This is a bug.")
    }
}

impl<OL: ObservableHList, O1: Observable> ObservableHList for (OL, (ObservableMeta, O1)) {
    type Append<O: Observable> = ((OL, (ObservableMeta, O1)), (ObservableMeta, O));

    const LEN: usize = 1 + OL::LEN;

    fn new() -> Self {
        (OL::new(), (ObservableMeta::new(), <O1 as Observable>::new()))
    }

    fn has<O: Observable>() -> bool { 
        TypeId::of::<O>() == TypeId::of::<O1>() || OL::has::<O>()
    }

    fn observe(&mut self, system: &SystemData) {
        <O1 as Observable>::observe(&mut self.1.1, system);
        OL::observe(&mut self.0, system);
    }

    fn autocorrelation<O: Observable>(&self) -> Option<f64> {
        if <(ObservableMeta, O1)>::has::<O>() {
            self.0.autocorrelation::<O>()
        } else {
            OL::autocorrelation::<O>(&self.0)
        }
    }

    fn history<O: Observable>(&self) -> &[O::Output] {
        if <(ObservableMeta, O1)>::has::<O>() {
            let cast = (&self.1.1 as &dyn Any)
                .downcast_ref::<O>()
                .expect("O1 != O, this is a bug. In (OL, O1) history");
            <O as Observable>::history(cast)
        } else {
            OL::history::<O>(&self.0)
        }
    }

    fn measured<O: Observable>(&self) -> Option<O::Output> {
        if <(ObservableMeta, O1)>::has::<O>() {
            <(ObservableMeta, O1)>::measured::<O>(&self.1)
        } else {
            OL::measured::<O>(&self.0)
        }
    }

    fn reserve(&mut self, n: usize) {
        <O1 as Observable>::reserve(&mut self.1.1, n);
        <OL as ObservableHList>::reserve(&mut self.0, n);
    }

    fn state<O: Observable>(&self) -> &O {
        if <(ObservableMeta, O1)>::has::<O>() {
            (self as &dyn Any).downcast_ref::<O>().expect("State cast failed. This is a bug.")
        } else {
            OL::state::<O>(&self.0)
        }
    }

    fn state_mut<O: Observable>(&mut self) -> &mut O {
        if <(ObservableMeta, O1)>::has::<O>() {
            (self as &mut dyn Any).downcast_mut::<O>().expect("State cast failed. This is a bug.")
        } else {
            OL::state_mut::<O>(&mut self.0)
        }
    }
}

/// Collects observables to create an [`ObservableStore`].
pub struct ObservableBuilder<Obs: ObservableHList> {
    _marker: PhantomData<Obs>
}

impl ObservableBuilder<()> {
    /// Creates a new empty observable builder.
    pub fn new() -> Self {
        ObservableBuilder { _marker: PhantomData }
    }
}

impl<Obs: ObservableHList> ObservableBuilder<Obs> {
    /// Determines whether the builder contains the given observable.
    pub fn has<O: Observable>(&self) -> bool {
        Obs::has::<O>()
    }

    /// Determines the amount of observables in the builder.
    pub const fn len(&self) -> usize {
        Obs::LEN
    }

    /// Adds an observable to the builder.
    pub fn with<O: Observable>(self) -> ObservableBuilder<Obs::Append<O>> {
        if self.has::<O>() {
            panic!("Observable \"{}\" has already been registered.", O::NAME);
        }

        ObservableBuilder {
            _marker: PhantomData
        }
    }

    /// Creates an observable storage from this builder.
    pub fn build(self) -> ObservableStore<Obs> {
        ObservableStore {
            storage: Obs::new(),
            phase: ObservablePhase::BurnIn
        }
    }
}

/// The phase the observables are currently in.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ObservablePhase {
    /// The system is still waiting for thermalisation.
    BurnIn,
    /// The system is collecting samples to compute the autocorrelation times.
    Sampling,
    /// The system is measuring.
    Normal
}

/// Stores observables and their associated data.
/// 
/// This uses the [`ObservableHList`] trait internally to store an arbitrary list of observable
/// types in a single struct. Unlike the naive dynamic dispatch approach of using `Box<dyn Any>`, this
/// method completely avoids dynamic dispatch by encoding the contents of the storage inside of the type at
/// compile time. This means that the compiler will evaluate a lot of the code at compile time rather than
/// runtime.
#[derive(Clone)]
pub struct ObservableStore<Obs: ObservableHList> {
    // Originally wanted to implement `Debug` for this to easily print out observable values but it seems
    // like this is impossible unless I require that every single observable is `Debug`.
    //
    // Treating `Debug` and non-`Debug` observables differently either requires specialisation
    // (which is highly unstable) or autoref specialisation, which is stable but only work with
    // concrete types, not the generics we are dealing with.
    storage: Obs,
    phase: ObservablePhase
}

impl<Obs: ObservableHList> ObservableStore<Obs> {
    pub(crate) fn set_phase(&mut self, phase: ObservablePhase) {
        self.phase = phase;
    }

    pub fn phase(&self) -> ObservablePhase {
        self.phase
    }

    /// Determines whether the storage contains the given observable.
    /// 
    /// This function is entirely evaluated at compile time.
    pub fn has<O: Observable>(&self) -> bool {
        Obs::has::<O>()
    }

    /// Determines the amount of observables in the storage.
    pub const fn len(&self) -> usize {
        Obs::LEN
    }

    /// Attemps to performs measurements for all observables.
    #[inline]
    pub fn observe_all(&mut self, data: &SystemData) {
        self.storage.observe(data);
    }

    /// Returns the computed autocorrelation time for the given observable.
    #[inline]
    pub fn autocorrelation<O: Observable>(&mut self) -> Option<f64> {
        self.storage.autocorrelation::<O>()
    }

    /// Reserves capacity for `n` additional measurements.
    #[inline]
    pub fn reserve(&mut self, n: usize) {
        self.storage.reserve(n);
    }

    /// Returns the measurement history for the given observables.
    ///
    /// This function panics if the given observable was not registered with the system at build time.
    #[inline]
    pub fn history<O: Observable>(&self) -> &[O::Output] {
        self.storage.history::<O>()
    }

    /// Returns the last measurement for the given observable.
    /// 
    /// This function returns `None` if the observable has produced no measurements yet.
    /// It panics if the given observable has not been registered with the system at build time.
    #[inline]
    pub fn measured<O: Observable>(&self) -> Option<O::Output> {
        self.storage.measured::<O>()
    }

    /// Retrieves the observable's state
    ///
    /// This function panics if the observable was not registered.
    #[inline]
    pub fn state<O: Observable>(&self) -> &O {
        self.storage.state::<O>()
    }

    /// Retrieves a mutable reference to the state.
    ///
    /// This function panics if the observable was not registered.
    #[inline]
    pub fn state_mut<O: Observable>(&mut self) -> &mut O {
        self.storage.state_mut::<O>()
    }
}

// impl<Obs: Clone + ObservableHList> Clone for ObservableStore<Obs> {
//     fn clone(&self) -> Self {
//         ObservableStore {
//             storage: self.storage.clone(),
//             phase: 
//         }
//     }
// }

/// Represents an observable. This allows a struct to make arbitrary measurements of the system.
/// 
/// After thermalisation, the system will automatically collect several measurements to compute the
/// autocorrelation time of the observable. When this is completed, the observable will only be measured
/// every set autocorrelation times.
pub trait Observable: 'static {
    /// The data type of the measurements of this observable.
    /// 
    /// *Note*: ideally this should be a small type since it is cloned when it is accessed.
    type Output: Copy + 'static;

    /// The name of the observable. This is used as the identifier in Prometheus metrics.
    /// 
    /// The name should not contain any spaces.
    const NAME: &'static str;

    /// Creates a new empty observable.
    fn new() -> Self;
    /// Reserves capacity for `n` additional measurements.
    fn reserve(&mut self, n: usize);
    /// Performs a new measurement. This measurement is then stored internally to possibly be accessed
    /// later.
    fn observe(&mut self, system: &SystemData);
    /// Access all measurements.
    fn history(&self) -> &[Self::Output];
    /// Access the latest measurement of the observable.
    fn latest(&self) -> Option<Self::Output>;
}