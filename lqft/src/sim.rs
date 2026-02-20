use crate::lattice::Lattice;
use crate::setup::{AcceptanceDesc, BurnInDesc};
use crate::snapshot::{SnapshotState};
use crate::stats::SystemStats;
use atomic_float::AtomicF64;
use hdf5_metno::H5Type;
use num_traits::Pow;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::{Xoshiro256PlusPlus, rand_core};
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use std::ops::Range;
use std::simd::num::SimdFloat;
use std::simd::Simd;
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool};
use std::time::Instant;
use crate::observable::{Observable, ObservableRegistry};
use crate::metrics::MetricState;

/// Makes all struct fields public in the current and specified modules.
/// This makes it easier to spread implementation details over multiple archive.
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

/// Simulation settings that should be saved to the archive.
#[derive(H5Type, Serialize, Deserialize)]
#[repr(C)]
pub struct SystemSettings {
    pub spacing: f64,
    pub step_size: f64,
    pub mass_squared: f64,
    pub coupling: f64,
}

impl From<SystemSettings> for [f64; 4] {
    fn from(settings: SystemSettings) -> [f64; 4] {
        [
            settings.spacing,
            settings.step_size,
            settings.mass_squared,
            settings.coupling,
        ]
    }
}

pub struct SystemData {
    pub lattice: Lattice,
    pub mass_squared: f64,
    pub coupling: f64,
    pub acceptance_desc: AcceptanceDesc,
    pub burn_in_desc: BurnInDesc,
    /// A vector for every possible C(t)
    /// where the inner vector is for every sweep
    pub correlation_slices: Vec<f64>,
    pub measurement_interval: usize,
    pub current_step_size: AtomicF64,
    pub stats: SystemStats,
}

all_public_in!(
    super,
    pub struct System {
        simulating: AtomicBool,
        metrics: MetricState,
        snapshot_state: Option<SnapshotState>,
        observables: ObservableRegistry,
        data: SystemData
    }
);

impl System {
    /// Simulates the system using the checkerboard method.
    ///
    /// The checkerboard method works by dividing the lattice into a "checkerboard" of "red" and "black" sites.
    /// Since many QFTs interact with direct neighbours only, this allows all sites of a single colour to be updated
    /// simultaneously rather than sequentially.
    ///
    /// We first divide the lattice into a colours using [`generate_checkerboard`](Lattice::generate_checkerboard_indices).
    /// Then, using a parallel iterator, we iterate over every single lattice of a given colour and attempt to flip it
    /// using the Metropolis algorithm.
    ///
    /// Internal safety:
    /// To make this work, the lattice makes use of interior mutability. While red is being simulated, all red sites are updated
    /// by independent threads (i.e. every red site is only accessed by a single thread at once). The black sites are not updated
    /// and are accessed by multiple threads at a time. The same holds for simulating black sites.
    pub fn simulate_checkerboard(&mut self, total_sweeps: usize) -> anyhow::Result<()> {
        self.data.stats.desired_sweeps = total_sweeps;

        if let Some(state) = &mut self.snapshot_state {
            state.init(self.data.lattice.dimensions(), total_sweeps)?;
        }

        self.data.stats.reserve_capacity(total_sweeps);
        self.data.correlation_slices
            .resize(self.data.lattice.dimensions()[0], 0.0);
        let (red, black) = self.data.lattice.generate_checkerboard_indices();

        tracing::info!("Running {total_sweeps} sweeps...");

        let mut sweep_timer;
        let mut total_timer = Instant::now();

        for i in 0..total_sweeps {
            sweep_timer = Instant::now();
            self.data.stats.current_sweep = i;

            // First update red sites....
            self.simulating.store(true, Ordering::SeqCst);
            
            let mut red_sites = unsafe { &mut *self.data.lattice.red_sites.get() };
            red_sites.par_iter_mut().for_each_init(
                || {
                    let mut seed_rng = rand::rng();
                    Xoshiro256PlusPlus::from_rng(&mut seed_rng)
                },
                |rng, site| {
                    
                }
            );
            
            let mut black_sites = unsafe { &mut *self.data.lattice.black_sites.get() };
            black_sites.par_iter_mut().for_each_init(
                || {
                    let mut seed_rng = rand::rng();
                    Xoshiro256PlusPlus::from_rng(&mut seed_rng)
                },
                |rng, site| {
                    
                }
            );
            
            self.simulating.store(false, Ordering::SeqCst);

            let sweep_time = sweep_timer.elapsed();
            sweep_timer = Instant::now();

            // Keep track of thermalisation ratio.
            let (th_ratio, thermalised) = self.compute_burn_in_ratio();
            if i > 2 * self.data.burn_in_desc.block_size {
                self.data.stats.thermalisation_ratio_history.push(th_ratio);
            }

            if self.data.stats.thermalised_at.is_none() && thermalised {
                tracing::info!("System has thermalised at sweep {i}");
                // System has thermalised, measurements can begin.
                self.data.stats.thermalised_at = Some(i);
            }

            // If system is thermalised and some amount of autocorrelation times have passed,
            // perform another measurement.
            if self.data.stats.thermalised_at.is_some() && i % self.data.measurement_interval == 0 {
                let sum_t = self.get_timeslice();
                let st = self.data.lattice.dimensions()[0];

                for t in 0..st {
                    let mut config_corr = 0.0;
                    for tp in 0..st {
                        let t_dist = (tp + t) % st;
                        config_corr += sum_t[tp] * sum_t[t_dist];
                    }

                    self.data.correlation_slices[t] += config_corr / (st as f64);
                }

                self.data.stats.performed_measurements += 1;
            }

            // Record statistics on every sweep.
            self.record_stats(sweep_time, &sweep_timer, i, total_sweeps)?;

            if total_timer.elapsed().as_secs() >= 1 {
                self.push_metrics();
                total_timer = Instant::now();

                tracing::info!("Sweep time: {}", sweep_time.as_micros());
            }

            self.observables.measure(&mut self.data);
        }

        self.push_metrics();

        for t in 0..self.data.lattice.dimensions()[0] {
            self.data.correlation_slices[t] /= self.data.stats.performed_measurements as f64;
        }

        tracing::info!("Run completed");

        Ok(())
    }

    /// Obtains the latest measurements of the given observable.
    #[inline]
    pub fn measured<O: Observable>(&self) -> Option<f64> {
        self.observables.measured::<O>()
    }

    pub fn observables(&self) -> &ObservableRegistry {
        &self.observables
    }

    pub fn observables_mut(&mut self) -> &mut ObservableRegistry {
        &mut self.observables
    }

    /// Computes the autocorrelation time of the system in its current state.
    fn compute_autocorrelation(&self) -> f64 {
        todo!()
    }

    /// Computes the absolute action of the entire lattice.
    pub fn compute_full_action(&self) -> f64 {
        let mut action = 0.0;
        for i in 0..self.data.lattice.sweep_size() {
            let a = self.data.lattice.spacing();

            let val = unsafe { *self.data.lattice[i].get() };
            let mut der_sum = 0.0;

            for j in 0..4 {
                // TODO: Create prebuilt adjacency table
                // let orig = self.lattice.from_index(i);
                let fneigh = self.data.lattice.get_forward_neighbor(i, j);
                let bneigh = self.data.lattice.get_backward_neighbor(i, j);

                // SAFETY: Neighbor sites will always only be read from due to checkerboarding.
                let fneigh_val = unsafe { *self.data.lattice[fneigh].get() };
                let bneigh_val = unsafe { *self.data.lattice[bneigh].get() };

                der_sum += ((fneigh_val - val) / a).pow(2) + ((val - bneigh_val) / a).pow(2);
            }

            let kinetic_delta = 0.5 * der_sum;
            let mass_delta = 0.5 * self.data.mass_squared * val.pow(2);
            let interaction_delta = 1.0 / 24.0 * self.data.coupling * val.pow(4);

            action += a.pow(4) * (kinetic_delta + mass_delta + interaction_delta);
        }

        action
    }

    fn get_timeslice(&mut self) -> Vec<f64> {
        // Compute the spatial sum for every time slice
        let st = self.data.lattice.dim_t();

        let mut sum_t = vec![0.0; st];
        for i in 0..self.data.lattice.sweep_size() {
            let t = self.data.lattice.from_index(i)[0];
            let val = unsafe { *self.data.lattice[i].get() };
            sum_t[t] += val;
        }

        sum_t
    }
}

/// Getters and setters
impl System {
    /// The current statistics of the simulation.
    pub fn stats(&self) -> &SystemStats {
        &self.data.stats
    }

    /// The current thermalisation ratio threshold.
    ///
    /// See [`SystemBuilder::th_threshold`] for more information.
    pub fn th_threshold(&self) -> f64 {
        self.data.burn_in_desc.required_ratio
    }

    pub fn current_step_size(&self) -> f64 {
        self.data.current_step_size.load(Ordering::Relaxed)
    }

    /// The current thermalisation block size.
    ///
    /// See [`SystemBuilder::th_block_size`](SystemBuilder::th_block_size) for more information.
    pub fn th_block_size(&self) -> usize {
        self.data.burn_in_desc.block_size
    }

    pub fn correlator2(&self) -> &[f64] {
        &self.data.correlation_slices
    }

    /// The current mass squared.
    pub fn mass_squared(&self) -> f64 {
        self.data.mass_squared
    }

    /// The current coupling constant.
    pub fn coupling(&self) -> f64 {
        self.data.coupling
    }

    /// The internal lattice used for data storage.
    pub fn lattice(&self) -> &Lattice {
        &self.data.lattice
    }

    /// Whether the system has thermalised yet.
    ///
    /// Once this returns true, the system is ready for measurement.
    pub fn thermalised(&self) -> bool {
        self.data.stats.thermalised_at.is_some()
    }
}
