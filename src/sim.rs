use std::ops::Range;
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
    dvar: f64,

    mass_squared: f64,
    bare_coupling: f64,
}

impl SimBuilder {
    pub fn new() -> Self {
        Self {
            spacing: 0.05, sizes: [100; 4], initial_value: InitialFieldValue::Fixed(0.0), dvar: 0.3, mass_squared: 1.0, bare_coupling: 0.0,
        }
    }

    pub fn spacing(mut self, value: f64) -> Self {
        self.spacing = value;
        self
    }

    pub fn sizes(mut self, value: [usize; 4]) -> Self {
        self.sizes = value;
        self
    }

    pub fn initial_value(mut self, value: InitialFieldValue) -> Self {
        self.initial_value = value;
        self
    }

    pub fn variation(mut self, value: f64) -> Self {
        self.dvar = value;
        self
    }

    pub fn bare_coupling(mut self, value: f64) -> Self {
        self.bare_coupling = value;
        self
    }

    pub fn mass_squared(mut self, value: f64) -> Self {
        self.mass_squared = value;
        self
    }

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
            lattice, spacing: self.spacing, dvar: self.dvar, mass_squared: self.mass_squared, coupling: self.bare_coupling,
            total_moves: 0, accepted_moves: 0
        };

        Ok(sim)
    }
}

impl Default for SimBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Sim {
    lattice: ScalarLattice4D,
    spacing: f64,
    dvar: f64,

    mass_squared: f64,
    coupling: f64,

    total_moves: usize,
    accepted_moves: usize
}

impl Sim {
    pub fn timestep(&mut self) {
        // Choose a random lattice site.
        let mut rng = rand::rng();

        let chosen_site = rng.random_range(0..self.lattice.sweep_size());
        self.perform_site_flip(chosen_site);

        // let curr_val = self.lattice[chosen_site];
        // let curr_action = self.perform_site_flip();
        //
        // let new_val = rng.random_range((curr_val - self.dvar)..(curr_val + self.dvar));
        // self.lattice[chosen_site] = new_val;
        // let new_action = self.perform_site_flip();
        //
        // let action_delta = curr_action - new_action;
        // let accept_prob = (-action_delta).exp();
        // println!("Accept prob: {accept_prob}");
        //
        // let realised = rng.random_range(0.0..1.0);
        // if realised > accept_prob {
        //     // Probability check failed.
        //     self.lattice[chosen_site] = curr_val;
        // }
    }

    pub fn total_moves(&self) -> usize { self.total_moves }
    pub fn accepted_moves(&self) -> usize { self.accepted_moves }

    pub fn lattice(&self) -> &ScalarLattice4D {
        &self.lattice
    }

    // Computes the action of the current configuration
    fn perform_site_flip(&mut self, site: usize) {
        let mut rng = rand::rng();
        let a = self.spacing;

        let curr_val = self.lattice[site];
        // println!("curr_val: {curr_val}");

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
            self.accepted_moves += 1;
            self.lattice[site] = new_val;
        }
        self.total_moves += 1;

        // println!("accept_prob: {accept_prob}");
        // println!("total_delta: {total_delta}");
    }
}
