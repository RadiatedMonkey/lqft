use crate::lattice::ScalarLattice4D;
use atomic_float::AtomicF64;
use num_traits::Pow;
use rand::Rng;
use std::ops::Range;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use rayon::prelude::*;

pub enum InitialFieldValue {
    Fixed(f64),
    RandomRange(Range<f64>),
}

pub struct SimBuilder {
    spacing: f64,
    sizes: [usize; 4],
    initial_value: InitialFieldValue,

    initial_step_size: f64,
    lower_acceptance: f64,
    upper_acceptance: f64,
    acceptance_update_interval: usize,

    mass_squared: f64,
    bare_coupling: f64,
}

impl SimBuilder {
    pub fn new() -> Self {
        Self {
            spacing: 0.05,
            sizes: [100; 4],
            initial_value: InitialFieldValue::Fixed(0.0),
            initial_step_size: 0.3,
            mass_squared: 1.0,
            bare_coupling: 0.0,
            lower_acceptance: 0.78,
            upper_acceptance: 0.82,
            acceptance_update_interval: 1000,
        }
    }

    /// Sets the lattice spacing.
    pub fn spacing(mut self, value: f64) -> Self {
        self.spacing = value;
        self
    }

    /// Sets the dimensions of the lattice.
    pub fn sizes(mut self, value: [usize; 4]) -> Self {
        self.sizes = value;
        self
    }

    /// Sets the initial conditions of the lattice.
    pub fn initial_value(mut self, value: InitialFieldValue) -> Self {
        self.initial_value = value;
        self
    }

    /// The starting value of the maximum field variation. The field variation determines
    /// how much the field fluctuates.
    pub fn initial_step_size(mut self, value: f64) -> Self {
        self.initial_step_size = value;
        self
    }

    /// Sets the desired lower bound on the acceptance ratio. If the acceptance ratio goes below
    /// this bound, dvar will be adjusted to correct this.
    pub fn lower_acceptance(mut self, ratio: f64) -> Self {
        self.lower_acceptance = ratio;
        self
    }

    /// Sets the desired upper bound on the acceptance ratio. If the acceptance ratio goes above
    /// this bound, dvar will be adjusted to correct this.
    pub fn upper_acceptance(mut self, ratio: f64) -> Self {
        self.upper_acceptance = ratio;
        self
    }

    /// Sets the amount of steps to wait before checking the field variation again.
    pub fn acceptance_update_interval(mut self, interval: usize) -> Self {
        self.acceptance_update_interval = interval;
        self
    }

    /// Sets the coupling constant.
    pub fn coupling(mut self, value: f64) -> Self {
        self.bare_coupling = value;
        self
    }

    /// Sets the mass squared. Setting this to a negative value will introduce symmetry breaking.
    pub fn mass_squared(mut self, value: f64) -> Self {
        self.mass_squared = value;
        self
    }

    /// Creates the simulation using the given options.
    pub fn build(self) -> anyhow::Result<Sim> {
        let lattice = match self.initial_value {
            InitialFieldValue::Fixed(val) => ScalarLattice4D::filled(self.sizes, val),
            InitialFieldValue::RandomRange(range) => ScalarLattice4D::random(self.sizes, range),
        };

        let mut sim = Sim {
            lattice,
            spacing: self.spacing,
            step_size: AtomicF64::new(self.initial_step_size),
            mass_squared: self.mass_squared,
            coupling: self.bare_coupling,
            lower_acceptance: self.lower_acceptance,
            upper_acceptance: self.upper_acceptance,
            stats: SimStatistics::default(),
            acceptance_interval: self.acceptance_update_interval,
        };

        let first_action = sim.compute_full_action();
        sim.stats
            .current_action
            .store(first_action, Ordering::Release);
        sim.stats.action_history.push(first_action);

        Ok(sim)
    }
}

impl Default for SimBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics of the simulation. Every finished sweep, a new statistic is recorded.
pub struct SimStatistics {
    /// Total moves made in the simulation
    pub total_moves: AtomicUsize,
    /// Total moves accepted
    pub accepted_moves: AtomicUsize,
    /// History of accepted moves. A new datapoint is recorded on every statistics interval.
    pub accepted_move_history: Vec<usize>,
    /// History of accepted move ratio.
    pub accept_ratio_history: Vec<f64>,
    /// History of the step size.
    pub dvar_history: Vec<f64>,
    /// History of the field mean
    pub mean_history: Vec<f64>,
    /// History of the field variance.
    pub meansq_history: Vec<f64>,
    /// History of the action over time.
    pub action_history: Vec<f64>,
    /// The action at the current point in time.
    pub current_action: AtomicF64,
}

impl SimStatistics {
    pub fn reserve_capacity(&mut self, count: usize) {
        self.accepted_move_history.reserve(count);
        self.accept_ratio_history.reserve(count);
        self.dvar_history.reserve(count);
        self.mean_history.reserve(count);
        self.meansq_history.reserve(count);
        self.action_history.reserve(count);
    }
}

impl Default for SimStatistics {
    fn default() -> Self {
        Self {
            current_action: AtomicF64::new(0.0),
            total_moves: AtomicUsize::new(0),
            accepted_move_history: Vec::new(),
            accepted_moves: AtomicUsize::new(0),
            accept_ratio_history: Vec::new(),
            dvar_history: Vec::new(),
            mean_history: Vec::new(),
            meansq_history: Vec::new(),
            action_history: Vec::new(),
        }
    }
}

pub struct Sim {
    lattice: ScalarLattice4D,
    spacing: f64,
    step_size: AtomicF64,

    mass_squared: f64,
    coupling: f64,

    acceptance_interval: usize,
    lower_acceptance: f64,
    upper_acceptance: f64,

    stats: SimStatistics,
}

impl Sim {
    pub fn stats(&self) -> &SimStatistics {
        &self.stats
    }

    pub fn step_size(&self) -> f64 {
        self.step_size.load(Ordering::Relaxed)
    }

    // /// Determines whether thermalisation of the system has finished.
    // fn has_thermalised(&self) -> bool {

    // }

    // Check whether the current field variation is correct, otherwise adjusts it slightly.
    fn update_step_size(&self) {
        let acceptance_ratio = self.accepted_moves() as f64 / self.total_moves() as f64;

        // Adjust dvar if acceptance ratio is 5% away from desired ratio
        if acceptance_ratio < self.lower_acceptance {
            let _ = self
                .step_size
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |f| Some(f * 0.95));
        } else if acceptance_ratio > self.upper_acceptance {
            let _ = self
                .step_size
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |f| Some(f * 1.05));
        }
    }

    /// Computes the absolute action of the entire lattice.
    fn compute_full_action(&self) -> f64 {
        let mut action = 0.0;
        for i in 0..self.lattice.sweep_size() {
            let a = self.spacing;

            let val = unsafe { *self.lattice[i].get() };
            let mut der_sum = 0.0;

            for j in 0..4 {
                // TODO: Create prebuilt adjacency table
                let orig = self.lattice.from_index(i);
                let fneigh = self.lattice.get_forward_neighbor(orig, j);
                let bneigh = self.lattice.get_backward_neighbor(orig, j);

                // SAFETY: Neighbor sites will always only be read from due to checkerboarding.
                let fneigh_val = unsafe { *self.lattice[fneigh].get() };
                let bneigh_val = unsafe { *self.lattice[bneigh].get() };

                der_sum += ((fneigh_val - val) / a).pow(2) + ((val - bneigh_val) / a).pow(2);
            }

            let kinetic_delta = 0.5 * der_sum;
            let mass_delta = 0.5 * self.mass_squared * val.pow(2);
            let interaction_delta = 1.0 / 24.0 * self.coupling * val.pow(4);

            action += a.pow(4) * (kinetic_delta + mass_delta + interaction_delta);
        }

        action
    }

    /// Performs a single iteration and returns the new field value at the given site.
    fn checkerboard_site_flip(&self, site: usize) -> f64 {
        let mut rng = rand::rng();
        let a = self.spacing;

        let curr_val = unsafe { *self.lattice[site].get() };

        let step_size = self.step_size();
        let new_val = rng.random_range((curr_val - step_size)..(curr_val + step_size));

        let mut curr_der_sum = 0.0;
        let mut new_der_sum = 0.0;

        for i in 0..4 {
            // TODO: Create prebuilt adjacency table
            let orig = self.lattice.from_index(site);
            let fneigh = self.lattice.get_forward_neighbor(orig, i);
            let bneigh = self.lattice.get_backward_neighbor(orig, i);

            // SAFETY: Neighbor sites will always only be read from due to checkerboarding.
            let fneigh_val = unsafe { *self.lattice[fneigh].get() };
            let bneigh_val = unsafe { *self.lattice[bneigh].get() };

            curr_der_sum +=
                ((fneigh_val - curr_val) / a).pow(2) + ((curr_val - bneigh_val) / a).pow(2);
            new_der_sum +=
                ((fneigh_val - new_val) / a).pow(2) + ((new_val - bneigh_val) / a).pow(2);
        }

        let kinetic_delta = 0.5 * (new_der_sum - curr_der_sum);
        let mass_delta = 0.5 * self.mass_squared * (new_val.pow(2) - curr_val.pow(2));
        let interaction_delta = 1.0 / 24.0 * self.coupling * (new_val.pow(4) - curr_val.pow(4));

        let total_delta: f64 = a.pow(4) * (kinetic_delta + mass_delta + interaction_delta);
        let accept_prob = (-total_delta).exp();
        let realised = rng.random_range(0.0..1.0);

        self.stats.total_moves.fetch_add(1, Ordering::Relaxed);
        if realised < accept_prob {
            self.stats
                .current_action
                .fetch_add(total_delta, Ordering::AcqRel);
            self.stats.accepted_moves.fetch_add(1, Ordering::Relaxed);
            return new_val;
        }

        curr_val
    }

    fn record_stats(&mut self, current_sweep: usize, total_sweeps: usize) {
        // Record statistics
        let mean = self.lattice.mean();
        let var = self.lattice.variance();
        let action = self.stats.current_action.load(Ordering::Acquire);

        self.stats.mean_history.push(mean);
        self.stats.meansq_history.push(var);
        self.stats.action_history.push(action);

        let accept = self.accepted_moves();
        let total = self.total_moves();
        let ratio = accept as f64 / total as f64 * 100.0;

        self.stats.accepted_move_history.push(self.accepted_moves());
        self.stats.accept_ratio_history.push(ratio);
        self.stats.dvar_history.push(self.step_size());

        println!("---------------------------------------------");
        println!(
            "Sweep progress ({:.2}%): {current_sweep}/{total_sweeps}",
            current_sweep as f64 / total_sweeps as f64 * 100.0
        );
        println!("Acceptance ratio: {:.2}%", ratio);
    }

    fn check_thermalisation(&self) -> bool {
        todo!()
    }

    pub fn simulate_checkerboard(&mut self, total_sweeps: usize) {
        let (red, black) = self.lattice.generate_indices();

        println!("Simulating using checkerboard method...");

        for i in 0..total_sweeps {
            // First update red sites....
            red.par_iter().for_each(|&index| {
                let new_site = self.checkerboard_site_flip(index);

                unsafe {
                    *self.lattice.sites[index].get() = new_site;
                }

                if index % self.acceptance_interval == 0 {
                    self.update_step_size();
                }
            });

            // then black sites.
            black.par_iter().for_each(|&index| {
                let new_site = self.checkerboard_site_flip(index);

                unsafe {
                    *self.lattice.sites[index].get() = new_site;
                }

                if index % self.acceptance_interval == 0 {
                    self.update_step_size();
                }
            });

            self.record_stats(i, total_sweeps);
            // let thermalised = self.check_thermalisation();
        }

        println!("Checkerboard simulation completed");
    }

    pub fn total_moves(&self) -> usize {
        self.stats.total_moves.load(Ordering::Relaxed)
    }
    pub fn accepted_moves(&self) -> usize {
        self.stats.accepted_moves.load(Ordering::Relaxed)
    }

    pub fn lattice(&self) -> &ScalarLattice4D {
        &self.lattice
    }
}
