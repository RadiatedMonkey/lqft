use std::ops::Range;
use std::time::Instant;
use num_traits::Pow;
use rand::Rng;
use crate::lattice::ScalarLattice4D;

pub enum InitialFieldValue {
    Fixed(f64),
    RandomRange(Range<f64>)
}

pub struct SimBuilder {
    spacing: f64,
    sizes: [usize; 4],
    initial_value: InitialFieldValue,

    starting_variation: f64,
    lower_acceptance: f64,
    upper_acceptance: f64,
    acceptance_update_interval: usize,

    mass_squared: f64,
    bare_coupling: f64,
    stats_interval: usize
}

impl SimBuilder {
    pub fn new() -> Self {
        Self {
            spacing: 0.05, sizes: [100; 4], initial_value: InitialFieldValue::Fixed(0.0), starting_variation: 0.3, mass_squared: 1.0, bare_coupling: 0.0,
            lower_acceptance: 0.78, upper_acceptance: 0.82, stats_interval: 100_000, acceptance_update_interval: 1000
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
    pub fn initial_variation(mut self, value: f64) -> Self {
        self.starting_variation = value;
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

    /// Decides how often to record statistics. The simulation will record a statistic every
    /// `interval` steps.
    pub fn stats_interval(mut self, interval: usize) -> Self {
        self.stats_interval = interval;
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
            InitialFieldValue::Fixed(val) => {
                ScalarLattice4D::filled(self.sizes, val)
            },
            InitialFieldValue::RandomRange(range) => {
                ScalarLattice4D::random(self.sizes, range)
            }
        };

        let sim = Sim {
            lattice, spacing: self.spacing, dvar: self.starting_variation, mass_squared: self.mass_squared, coupling: self.bare_coupling,
            lower_acceptance: self.lower_acceptance, upper_acceptance: self.upper_acceptance, stats: SimStatistics::default(), stats_interval: self.stats_interval, acceptance_interval: self.acceptance_update_interval
        };

        Ok(sim)
    }
}

impl Default for SimBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SimStatistics {
    pub total_moves: usize,
    pub accepted_moves: usize,
    pub accept_ratio_history: Vec<f64>,
    pub dvar_history: Vec<f64>,
    pub mean_history: Vec<f64>,
    pub var_history: Vec<f64>
}

impl SimStatistics {
    pub fn reserve_capacity(&mut self, count: usize) {
        self.accept_ratio_history.reserve(count);
        self.dvar_history.reserve(count);
        self.mean_history.reserve(count);
        self.var_history.reserve(count);
    }
}

impl Default for SimStatistics {
    fn default() -> Self {
        Self {
            total_moves: 0,
            accepted_moves: 0,
            accept_ratio_history: Vec::with_capacity(100),
            dvar_history: Vec::with_capacity(100),
            mean_history: Vec::with_capacity(100),
            var_history: Vec::with_capacity(100)
        }
    }
}

pub struct Sim {
    lattice: ScalarLattice4D,
    spacing: f64,
    dvar: f64,

    mass_squared: f64,
    coupling: f64,

    acceptance_interval: usize,
    lower_acceptance: f64,
    upper_acceptance: f64,

    stats_interval: usize,
    stats: SimStatistics
}

impl Sim {

    pub fn stats(&self) -> &SimStatistics {
        &self.stats
    }

    pub fn simulate(&mut self, total_sweeps: usize) {
        let sweep = self.lattice.sweep_size();
        let step_count = total_sweeps * sweep;

        let stats_count = step_count / self.stats_interval;
        self.stats.reserve_capacity(stats_count);

        for i in 0..step_count {
            self.timestep();

            // Record statistics
            if i % self.stats_interval == 0 {
                let mean = self.lattice.mean();
                let var = self.lattice.variance();

                self.stats.mean_history.push(mean);
                self.stats.var_history.push(var);

                let accept = self.stats.accepted_moves;
                let total = self.stats.total_moves;
                let ratio = accept as f64 / total as f64 * 100.0;

                self.stats.accept_ratio_history.push(ratio);
                self.stats.dvar_history.push(self.dvar);

                println!("---------------------------------------------");
                println!("Progress ({:.2}%): {i}/{step_count}", i as f64 / step_count as f64 * 100.0);
                println!("Acceptance ratio: {:.2}%", ratio);
            }

            // Update field variation
            if i % self.acceptance_interval == 0 {
                self.update_dvar();
            }
        }
    }

    fn timestep(&mut self) {
        // Choose a random lattice site.
        let mut rng = rand::rng();
        let chosen_site = rng.random_range(0..self.lattice.sweep_size());
        self.perform_site_flip(chosen_site);
    }

    pub fn total_moves(&self) -> usize { self.stats.total_moves }
    pub fn accepted_moves(&self) -> usize { self.stats.accepted_moves }

    pub fn lattice(&self) -> &ScalarLattice4D {
        &self.lattice
    }

    // Computes the action of the current configuration
    fn perform_site_flip(&mut self, site: usize) {
        let mut rng = rand::rng();
        let a = self.spacing;

        let curr_val = self.lattice[site];
        let new_val = rng.random_range((curr_val - self.dvar)..(curr_val + self.dvar));

        let mut curr_der_sum = 0.0;
        let mut new_der_sum = 0.0;

        for i in 0..4 {
            // TODO: Create prebuilt adjacency table
            let orig = self.lattice.from_index(site);
            let fneigh = self.lattice.get_forward_neighbor(orig, i);
            let bneigh = self.lattice.get_backward_neighbor(orig, i);

            let fneigh_val = self.lattice[fneigh];
            let bneigh_val = self.lattice[bneigh];

            curr_der_sum += ((fneigh_val - curr_val) / a).pow(2) + ((curr_val - bneigh_val) / a).pow(2);
            new_der_sum += ((fneigh_val - new_val) / a).pow(2) + ((new_val - bneigh_val) / a).pow(2);
        }

        let kinetic_delta = 0.5 * (new_der_sum - curr_der_sum);
        let mass_delta = 0.5 * self.mass_squared * (new_val.pow(2) - curr_val.pow(2));
        let interaction_delta = 1.0 / 24.0 * self.coupling * (new_val.pow(4) - curr_val.pow(4));

        let total_delta: f64 = a.pow(4) * (kinetic_delta + mass_delta + interaction_delta);
        let accept_prob = (-total_delta).exp();
        let realised = rng.random_range(0.0..1.0);
        if realised < accept_prob {
            self.stats.accepted_moves += 1;
            self.lattice[site] = new_val;
        }
        self.stats.total_moves += 1;
    }

    // Check whether the current field variation is correct, otherwise adjusts it slightly.
    fn update_dvar(&mut self) {
        let acceptance_ratio = self.stats.accepted_moves as f64 / self.stats.total_moves as f64;

        // Adjust dvar if acceptance ratio is 5% away from desired ratio
        if acceptance_ratio < self.lower_acceptance {
            self.dvar *= 0.95;
        } else if acceptance_ratio > self.upper_acceptance {
            self.dvar *= 1.05;
        }
    }
}
