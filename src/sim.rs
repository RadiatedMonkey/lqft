use crate::lattice::ScalarLattice4D;
use atomic_float::AtomicF64;
use num_traits::Pow;
use rand::Rng;
use std::ops::Range;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use rayon::prelude::*;

/// Makes all struct fields public in the current and specified modules.
/// This makes it easier to spread implementation details over multiple files.
macro_rules! all_public_in {
    ($module:path, $vis:vis struct $name:ident {
        $(
            $(#[$meta:meta])*
            $field_name:ident: $field_type:ty
        ),* $(,)?
    }) => {
        $vis struct $name {
            $(
                $(#[$meta])*
                pub(in $module) $field_name : $field_type
            ),*
        }
    }
}

/// The method of initialising the lattice.
pub enum InitialState {
    /// Sets every lattice site to the same fixed value.
    Fixed(f64),
    /// Randomises the lattice using values from this range.
    RandomRange(Range<f64>),
}

/// Used to configure a lattice simulation.
pub struct SystemBuilder {
    spacing: f64,
    sizes: [usize; 4],
    initial_state: InitialState,

    initial_step_size: f64,
    lower_acceptance: f64,
    upper_acceptance: f64,
    step_size_correction: f64,

    acceptance_update_interval: usize,
    thermalisation_threshold: f64,
    thermalisation_block_size: usize,

    mass_squared: f64,
    bare_coupling: f64,
}

impl SystemBuilder {
    pub fn new() -> Self {
        Self {
            spacing: 1.0,
            sizes: [40, 20, 20, 20],
            initial_state: InitialState::Fixed(0.0),
            initial_step_size: 1.0,
            mass_squared: 1.0,
            bare_coupling: 0.0,
            lower_acceptance: 0.3,
            upper_acceptance: 0.5,
            step_size_correction: 0.05,
            thermalisation_threshold: 0.01,
            thermalisation_block_size: 100,
            acceptance_update_interval: 1000
        }
    }

    /// Sets the change in step size for each correction.
    /// 
    /// If the system determines the acceptance ratio to be outside of the desired range,
    /// it may increase or decrease the step size by this ratio.
    /// 
    /// Default value: `0.05`
    pub fn step_size_correction(mut self, value: f64) -> Self {
        self.step_size_correction = value;
        self
    }

    /// Sets the lattice spacing.
    /// 
    /// Default value: `1.0`.
    pub fn spacing(mut self, value: f64) -> Self {
        self.spacing = value;
        self
    }

    /// Sets the dimensions of the lattice.
    /// 
    /// These dimensions are in the format `[t, x, y, z]`.
    /// 
    /// Default value: `[40, 20, 20, 20]`
    pub fn sizes(mut self, value: [usize; 4]) -> Self {
        self.sizes = value;
        self
    }

    /// Sets the initial state of the lattice.
    /// 
    /// The initial state can either be set to a constant value (for example `0.0` for a "cold start")
    /// or a random range for a "hot start".
    /// 
    /// Default value: `LatticeInitialization::FixedValue(0.0)`.
    pub fn initial_value(mut self, value: InitialState) -> Self {
        self.initial_state = value;
        self
    }

    /// The starting value of the step size.
    /// 
    /// This should be set around the optimal step size for the current configuration (if known).
    /// If the initial step size is suboptimal, the system will automatically adjust it to reach the desired
    /// acceptance ratio. This may however cause thermalisation to take longer.
    /// 
    /// Default value: `1.0`.
    pub fn initial_step_size(mut self, value: f64) -> Self {
        self.initial_step_size = value;
        self
    }

    /// Sets the desired lower bound on the acceptance ratio. If the acceptance ratio goes below
    /// this bound, the step size will be adjusted to correct this.
    ///
    /// Default value: `0.3`.
    /// 
    /// See [`step_size_decrease`](SimBuilder::step_size_decrease) and [`initial_step_size`](SimBuilder::initial_step_size) for more information.
    pub fn lower_acceptance(mut self, ratio: f64) -> Self {
        self.lower_acceptance = ratio;
        self
    }

    /// Sets the desired upper bound on the acceptance ratio. If the acceptance ratio goes above
    /// this bound, the step size will be adjusted to correct this.
    /// 
    /// Default value: `0.5`.
    /// 
    /// See [`step_size_increase`](SimBuilder::step_size_increase) and [`initial_step_size`](SimBuilder::initial_step_size) for more information.
    pub fn upper_acceptance(mut self, ratio: f64) -> Self {
        self.upper_acceptance = ratio;
        self
    }

    /// Sets the amount of iterations before checking and correcting the step size.
    /// 
    /// Rather than correcting the step size every single iteration, the system checks it on a predetermined interval.
    /// This is to improve performance since most times only a few corrections are needed to steer the system to the right acceptance ratio.
    /// 
    /// Default value: `1000`.
    pub fn step_size_correction_interval(mut self, interval: usize) -> Self {
        self.acceptance_update_interval = interval;
        self
    }

    /// Sets the coupling constant.
    /// 
    /// This sets the strength of the 4phi coupling. If set to `0.0` the system will resemble a free scalar field.
    /// 
    /// Default value: `0.0`.
    pub fn coupling(mut self, value: f64) -> Self {
        self.bare_coupling = value;
        self
    }

    /// Sets the mass squared. Setting this to a negative value will introduce symmetry breaking.
    /// 
    /// Default value: `1.0`.
    pub fn mass_squared(mut self, value: f64) -> Self {
        self.mass_squared = value;
        self
    }   

    /// Sets the thermalisation threshold. See [`th_ratio`](System::th_ratio) for more information on how thermalisation
    /// works.
    /// 
    /// Default value: `0.01`.
    pub fn th_threshold(mut self, value: f64) -> Self {
        self.thermalisation_threshold = value;
        self
    }

    /// Sets the thermalisation block size. See [`th_ratio`](System::th_ratio) for more information on how thermalisation
    /// works.
    /// 
    /// Default value: `100`.
    pub fn th_block_size(mut self, value: usize) -> Self {
        self.thermalisation_block_size = value;
        self
    }

    /// Creates the simulation using the given options.
    pub fn build(self) -> anyhow::Result<System> {
        let lattice = match self.initial_state {
            InitialState::Fixed(val) => ScalarLattice4D::filled(self.sizes, val),
            InitialState::RandomRange(range) => ScalarLattice4D::random(self.sizes, range),
        };

        let mut sim = System {
            lattice,
            spacing: self.spacing,
            step_size: AtomicF64::new(self.initial_step_size),
            mass_squared: self.mass_squared,
            coupling: self.bare_coupling,
            lower_acceptance: self.lower_acceptance,
            upper_acceptance: self.upper_acceptance,
            step_size_correction: self.step_size_correction,
            stats: SystemStats::default(),
            step_size_correction_interval: self.acceptance_update_interval,
            th_block_size: self.thermalisation_block_size,
            th_threshold: self.thermalisation_threshold,
            correlation_slices: Vec::new(),
            measurement_interval: 50
        };

        let first_action = sim.compute_full_action();
        sim.stats
            .current_action
            .store(first_action, Ordering::Release);

        sim.stats.action_history.push(first_action);

        Ok(sim)
    }
}

impl Default for SystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics of the simulation. Every finished sweep, a new statistic is recorded.
pub struct SystemStats {
    /// Total moves made in the simulation
    pub total_moves: AtomicUsize,
    /// Total moves accepted
    pub accepted_moves: AtomicUsize,
    /// History of accepted moves. A new datapoint is recorded on every statistics interval.
    pub accepted_move_history: Vec<usize>,
    /// History of accepted move ratio.
    pub accept_ratio_history: Vec<f64>,
    /// History of the step size.
    pub step_size_history: Vec<f64>,
    /// History of the field mean
    pub mean_history: Vec<f64>,
    /// History of the field variance.
    pub meansq_history: Vec<f64>,
    /// History of the action over time.
    pub action_history: Vec<f64>,
    /// The action at the current point in time.
    pub current_action: AtomicF64,
    /// The history of the thermalisation ratio.
    pub thermalisation_ratio_history: Vec<f64>,
    /// The sweep at which the system first passed the thermalisation threshold.
    pub thermalised_at: Option<usize>,
    /// The amount of measurements performed after thermalisation.
    pub performed_measurements: usize
}

impl SystemStats {    
    /// Reserves enough space for `count` additional statistics.
    /// 
    /// This is called at the start of the simulation with the desired sweep count to improve performance.
    pub fn reserve_capacity(&mut self, count: usize) {
        self.accepted_move_history.reserve(count);
        self.accept_ratio_history.reserve(count);
        self.step_size_history.reserve(count);
        self.mean_history.reserve(count);
        self.meansq_history.reserve(count);
        self.action_history.reserve(count);
        self.thermalisation_ratio_history.reserve(count);
    }

    /// The most recent value of the whole system action.
    pub fn current_action(&self) -> f64 {
        self.current_action.load(Ordering::Relaxed)
    }

    /// The total amount of attempted moves so far.
    pub fn total_moves(&self) -> usize {
        self.total_moves.load(Ordering::Relaxed)
    }

    /// The total amount of accepted moves so far.
    pub fn accepted_moves(&self) -> usize {
        self.accepted_moves.load(Ordering::Relaxed)
    }
}

impl Default for SystemStats {
    fn default() -> Self {
        Self {
            current_action: AtomicF64::new(0.0),
            total_moves: AtomicUsize::new(0),
            accepted_move_history: Vec::new(),
            accepted_moves: AtomicUsize::new(0),
            accept_ratio_history: Vec::new(),
            step_size_history: Vec::new(),
            mean_history: Vec::new(),
            meansq_history: Vec::new(),
            action_history: Vec::new(),
            thermalisation_ratio_history: Vec::new(),
            thermalised_at: None,
            performed_measurements: 0
        }
    }
}

all_public_in!(super, pub struct System {
    lattice: ScalarLattice4D,
    spacing: f64,
    step_size: AtomicF64,

    mass_squared: f64,
    coupling: f64,

    step_size_correction_interval: usize,
    step_size_correction: f64,
    lower_acceptance: f64,
    upper_acceptance: f64,

    th_threshold: f64,
    th_block_size: usize,

    /// A vector for every possible C(t)
    /// where the inner vector is for every sweep
    correlation_slices: Vec<f64>,
    measurement_interval: usize,

    stats: SystemStats,
});

impl System {
    // ==============================================================================
    // Getters
    // ==============================================================================

    /// The current statistics of the simulation.
    pub fn stats(&self) -> &SystemStats {
        &self.stats
    }

    /// The lattice spacing.
    pub fn spacing(&self) -> f64 {
        self.spacing
    }

    /// The lower bound on the acceptance ratio.
    pub fn lower_acceptance(&self) -> f64 {
        self.lower_acceptance
    }

    /// The upper bound on the acceptance ratio.
    pub fn upper_acceptance(&self) -> f64 {
        self.upper_acceptance
    }

    pub fn step_size_correction_interval(&self) -> usize {
        self.step_size_correction_interval
    }

    /// The current thermalisation ratio threshold.
    /// 
    /// See [`SystemBuilder::th_threshold`] for more information.
    pub fn th_threshold(&self) -> f64 {
        self.th_threshold
    }

    /// The current thermalisation block size.
    /// 
    /// See [`SystemBuilder::th_block_size`](SystemBuilder::th_block_size) for more information.
    pub fn th_block_size(&self) -> usize {
        self.th_block_size
    }

    pub fn step_size(&self) -> f64 {
        self.step_size.load(Ordering::Relaxed)
    }

    /// Sets the new step size. 
    /// 
    /// This function should usually not be called directly. Instead let the system
    /// configure the step size by itself.
    pub fn set_step_size(&self, value: f64) {
        self.step_size.store(value, Ordering::Relaxed);
    }

    /// Gives the current step size correction.
    /// 
    /// See [`SystemBuilder::step_size_correction`](SystemBuilder::step_size_correction) for more information.
    pub fn step_size_correction(&self) -> f64 {
        self.step_size_correction
    }

    pub fn correlator2(&self) -> &[f64] {
        &self.correlation_slices
    }   

    /// The current mass squared.
    pub fn mass_squared(&self) -> f64 {
        self.mass_squared
    }

    /// The current coupling constant.
    pub fn coupling(&self) -> f64 {
        self.coupling
    }

    /// The internal lattice used for data storage.
    pub fn lattice(&self) -> &ScalarLattice4D {
        &self.lattice
    }

    /// Whether the system has thermalised yet.
    /// 
    /// Once this returns true, the system is ready for measurement.
    pub fn thermalised(&self) -> bool {
        self.stats.thermalised_at.is_some()
    }
    
    /// Computes the autocorrelation time of the system in its current state.
    fn compute_autocorrelation(&self) -> f64 {
        todo!()
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
    /// 
    /// SAFETY: This method should only be called if the calling thread has exclusive access to the given site.
    /// and the direct neighbours can safely be read from.
    unsafe fn checkerboard_site_flip(&self, site: usize) -> f64 {
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

    /// Records statistics on the current sweep.
    fn record_stats(&mut self, current_sweep: usize, total_sweeps: usize) {
        let mean = self.lattice.mean();
        let var = self.lattice.variance();
        let action = self.stats.current_action.load(Ordering::Acquire);

        self.stats.mean_history.push(mean);
        self.stats.meansq_history.push(var);
        self.stats.action_history.push(action);

        let accept = self.stats.accepted_moves();
        let total = self.stats.total_moves();
        let ratio = accept as f64 / total as f64 * 100.0;

        self.stats.accepted_move_history.push(accept);
        self.stats.accept_ratio_history.push(ratio);
        self.stats.step_size_history.push(self.step_size());

        // TODO: Create proper progress bar.
        println!(
            "Sweep progress ({:.2}%): {current_sweep}/{total_sweeps}",
            current_sweep as f64 / total_sweeps as f64 * 100.0
        );
        println!("Acceptance ratio: {:.2}%", ratio);
    }

    fn get_timeslice(&mut self) -> Vec<f64> {
        // Compute the spatial sum for every time slice
        let st = self.lattice.t_dim();

        let mut sum_t = vec![0.0; st];
        for i in 0..self.lattice.sweep_size() {
            let t = self.lattice.from_index(i)[0];
            let val = unsafe { *self.lattice[i].get() };
            sum_t[t] += val;
        }

        sum_t
    }

    /// Simulates the system using the checkerboard method.
    /// 
    /// The checkerboard method works by dividing the lattice into a "checkerboard" of "red" and "black" sites.
    /// Since many QFTs interact with direct neighbours only, this allows all sites of a single colour to be updated
    /// simultaneously rather than sequentially.
    /// 
    /// We first divide the lattice into a colours using [`generate_checkerboard`](ScalarLattice4D::generate_checkerboard).
    /// Then, using a parallel iterator, we iterate over every single lattice of a given colour and attempt to flip it
    /// using the Metropolis algorithm. 
    /// 
    /// Internal safety:
    /// To make this work, the lattice makes use of interior mutability. While red is being simulated, all red sites are updated
    /// by independent threads (i.e. every red site is only accessed by a single thread at once). The black sites are not updated
    /// and are accessed by multiple threads at a time. The same holds for simulating black sites.
    pub fn simulate_checkerboard(&mut self, total_sweeps: usize) {
        self.stats.reserve_capacity(total_sweeps);
        self.correlation_slices.resize(self.lattice.dimensions()[0], 0.0);
        let (red, black) = self.lattice.generate_checkerboard();

        println!("Simulating {total_sweeps} sweeps using checkerboard method...");
        
        for i in 0..total_sweeps {
            // First update red sites....
            red.par_iter().for_each(|&index| {
                // SAFETY: Since this is a red site, this thread has exclusive access to the site.
                // Therefore it can safely update the value.
                unsafe {
                    let new_site = self.checkerboard_site_flip(index);
                    *self.lattice.sites[index].get() = new_site;
                };

                if index % self.step_size_correction_interval == 0 && !self.thermalised() {
                    self.correct_step_size();
                }
            });

            // then black sites.
            black.par_iter().for_each(|&index| {
                // SAFETY: Since this is a black site, this thread has exclusive access to the site.
                // Therefore it can safely update the value.
                unsafe {
                    let new_site = self.checkerboard_site_flip(index);
                    *self.lattice.sites[index].get() = new_site;
                }

                if index % self.step_size_correction_interval == 0 && !self.thermalised() {
                    self.correct_step_size();
                }
            });

            // Keep track of thermalisation ratio.
            let (th_ratio, thermalised) = self.th_ratio();
            if i > 2 * self.th_block_size {
                self.stats.thermalisation_ratio_history.push(th_ratio);   
            }

            // Record statistics on every sweep.
            self.record_stats(i, total_sweeps);
            
            if self.stats.thermalised_at.is_none() && thermalised {
                // System has thermalised, measurements can begin.
                self.stats.thermalised_at = Some(i);
            }

            // If system is thermalised and some amount of autocorrelation times have passed,
            // perform another measurement.
            if self.stats.thermalised_at.is_some() && i % self.measurement_interval == 0 {
                let sum_t = self.get_timeslice();
                let st = self.lattice.dimensions()[0];

                for t in 0..st {
                    let mut config_corr = 0.0;
                    for tp in 0..st {
                        let t_dist = (tp + t) % st;
                        config_corr += sum_t[tp] * sum_t[t_dist];
                    }

                    self.correlation_slices[t] += config_corr / (st as f64);
                }

                self.stats.performed_measurements += 1;
            }

            // let thermalised = self.check_thermalisation();
        }

        for t in 0..self.lattice.dimensions()[0] {
            self.correlation_slices[t] /= self.stats.performed_measurements as f64;
        }

        println!("Checkerboard simulation completed");
    }
}
