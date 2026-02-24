//! Implements the observable API.

use std::any::{Any, TypeId};
use std::marker::PhantomData;
use crate::sim::SystemData;

pub trait ObservableHList {
    fn new() -> Self;
    fn has<O: Observable>() -> bool;
    fn observe(&mut self, system: &SystemData);
    fn measured<O: Observable>(&self) -> Option<O::Output>;
    fn reserve(&mut self, n: usize);
}

impl ObservableHList for () {
    fn new() -> () {}

    fn has<O: Observable>() -> bool { 
        false 
    }

    fn observe(&mut self, _system: &SystemData) {}

    fn measured<O: Observable>(&self) -> Option<O::Output> {
        panic!("Observable {} is not registered.", O::NAME);
    }

    fn reserve(&mut self, _n: usize) {}
}

impl<O1: Observable> ObservableHList for O1 {
    fn new() -> O1 { <O1 as Observable>::new() }

    fn has<O: Observable>() -> bool { 
        TypeId::of::<O>() == TypeId::of::<O1>() 
    }

    fn observe(&mut self, system: &SystemData) {
        <O1 as Observable>::observe(self, system);
    }

    fn measured<O: Observable>(&self) -> Option<O::Output> {
        if !Self::has::<O>() {
            None
        } else {
            let latest = self.latest();

            assert_eq!(TypeId::of::<O1::Output>(), TypeId::of::<O::Output>(), "Measured and requested type are not equal. This is a bug.");
            
            let any = latest
                .as_ref()
                .map(|x| (x as &dyn Any).downcast_ref::<O::Output>())
                .flatten();

            any.cloned()
        }
    }

    fn reserve(&mut self, n: usize) {
        <Self as Observable>::reserve(self, n);
    }
}

impl<OL: ObservableHList, O1: Observable> ObservableHList for (OL, O1) {
    fn new() -> (OL, O1) {
        (OL::new(), <O1 as Observable>::new())
    }

    fn has<O: Observable>() -> bool { 
        TypeId::of::<O>() == TypeId::of::<O1>() || OL::has::<O>()
    }

    fn observe(&mut self, system: &SystemData) {
        <O1 as Observable>::observe(&mut self.1, system);
        OL::observe(&mut self.0, system);
    }

    fn measured<O: Observable>(&self) -> Option<O::Output> {
        if O1::has::<O>() {
            O1::measured::<O>(&self.1)
        } else {
            OL::measured::<O>(&self.0)
        }
    }

    fn reserve(&mut self, n: usize) {
        <O1 as Observable>::reserve(&mut self.1, n);
        <OL as ObservableHList>::reserve(&mut self.0, n);
    }
}

pub struct ObservableBuilder<Obs: ObservableHList> {
    _marker: PhantomData<Obs>
}

impl ObservableBuilder<()> {
    pub fn new() -> Self {
        ObservableBuilder { _marker: PhantomData }
    }
}

impl<Obs: ObservableHList> ObservableBuilder<Obs> {
    pub fn has<O: Observable>(&self) -> bool {
        Obs::has::<O>()
    }

    pub fn with<O: Observable>(self) -> ObservableBuilder<(Obs, O)> {
        ObservableBuilder {
            _marker: PhantomData
        }
    }

    pub fn build(self) -> ObservableStorage<Obs> {
        ObservableStorage {
            storage: Obs::new()
        }
    }
}

pub struct ObservableStorage<Obs> {
    storage: Obs
}

impl<Obs: ObservableHList> ObservableStorage<Obs> {
    pub fn has<O: Observable>(&self) -> bool {
        Obs::has::<O>()
    }

    pub fn observe_all(&mut self, data: &SystemData) {
        self.storage.observe(data);
    }

    pub fn measured<O: Observable>(&self) -> Option<O::Output> {
        self.storage.measured::<O>()
    }
}

pub trait Observable: 'static {
    type Output: Clone + 'static;

    const NAME: &'static str;

    fn new() -> Self;
    fn reserve(&mut self, n: usize);
    fn observe(&mut self, system: &SystemData);
    fn latest(&self) -> Option<Self::Output>;
}