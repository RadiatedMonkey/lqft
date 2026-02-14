use crate::lattice::{AccessToken, ScalarLattice4D};
use atomic_float::AtomicF64;
use hdf5_metno::H5Type;
use num_traits::Pow;
use rand::Rng;
use serde::Deserialize;
use serde::Serialize;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread;
use std::thread::{JoinHandle, Thread};
use rayon::prelude::*;
use hdf5_metno as hdf5;
use ndarray::ArrayView5;
use crate::setup::{AcceptanceDesc, BurnInDesc};
use crate::snapshot::{SnapshotFragment, SnapshotState};
use crate::stats::SystemStats;

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
    pub coupling: f64
}

impl From<SystemSettings> for [f64; 4] {
    fn from(settings: SystemSettings) -> [f64; 4] {
        [settings.spacing, settings.step_size, settings.mass_squared, settings.coupling]
    }
}

all_public_in!(super, pub struct System {
    /// Whether the lattice is currently in use by the checkerboard method.
    /// If this is true, the system should not be read from.
    simulating: AtomicBool,

    lattice: ScalarLattice4D,

    mass_squared: f64,
    coupling: f64,

    acceptance_desc: AcceptanceDesc,
    burn_in_desc: BurnInDesc,

    /// A vector for every possible C(t)
    /// where the inner vector is for every sweep
    correlation_slices: Vec<f64>,
    measurement_interval: usize,

    current_step_size: AtomicF64,
    current_sweep: usize,

    snapshot_state: Option<SnapshotState>,
    stats: SystemStats,
});

impl System {
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
    pub fn simulate_checkerboard(&mut self, total_sweeps: usize) -> anyhow::Result<()> {
        if let Some(state) = &mut self.snapshot_state {
            state.init(self.lattice.dimensions(), total_sweeps)?;
        }

        self.stats.reserve_capacity(total_sweeps);
        self.correlation_slices.resize(self.lattice.dimensions()[0], 0.0);
        let (red, black) = self.lattice.generate_checkerboard();

        println!("Simulating {total_sweeps} sweeps using checkerboard method...");

        for i in 0..total_sweeps {
            self.current_sweep = i;

            // First update red sites....
            self.simulating.store(true, Ordering::SeqCst);
            red.par_iter().for_each(|&index| {
                // SAFETY: Since this is a red site, this thread has exclusive access to the site.
                // Therefore it can safely update the value.
                unsafe {
                    let new_site = self.checkerboard_site_flip(index);
                    *self.lattice[index].get() = new_site;
                };

                if index % self.acceptance_desc.correction_interval == 0 && !self.thermalised() {
                    self.correct_step_size();
                }
            });

            // then black sites.
            black.par_iter().for_each(|&index| {
                // SAFETY: Since this is a black site, this thread has exclusive access to the site.
                // Therefore it can safely update the value.
                unsafe {
                    let new_site = self.checkerboard_site_flip(index);
                    *self.lattice[index].get() = new_site;
                }

                if index % self.acceptance_desc.correction_interval == 0 && !self.thermalised() {
                    self.correct_step_size();
                }
            });
            self.simulating.store(false, Ordering::SeqCst);

            // Keep track of thermalisation ratio.
            let (th_ratio, thermalised) = self.compute_burn_in_ratio();
            if i > 2 * self.burn_in_desc.avg_block_size {
                self.stats.thermalisation_ratio_history.push(th_ratio);
            }

            // Record statistics on every sweep.
            self.record_stats()?;

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

            println!("Sweep {i}/{total_sweeps}");
        }

        for t in 0..self.lattice.dimensions()[0] {
            self.correlation_slices[t] /= self.stats.performed_measurements as f64;
        }

        println!("Checkerboard simulation completed");

        Ok(())
    }

    /// Computes the autocorrelation time of the system in its current state.
    fn compute_autocorrelation(&self) -> f64 {
        todo!()
    }

    /// Computes the absolute action of the entire lattice.
    pub fn compute_full_action(&self) -> f64 {
        let mut action = 0.0;
        for i in 0..self.lattice.sweep_size() {
            let a = self.lattice.spacing();

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
        let a = self.lattice.spacing();

        let curr_val = unsafe { *self.lattice[site].get() };

        let step_size = self.current_step_size();
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

    fn get_timeslice(&mut self) -> Vec<f64> {
        // Compute the spatial sum for every time slice
        let st = self.lattice.dim_t();

        let mut sum_t = vec![0.0; st];
        for i in 0..self.lattice.sweep_size() {
            let t = self.lattice.from_index(i)[0];
            let val = unsafe { *self.lattice[i].get() };
            sum_t[t] += val;
        }

        sum_t
    }
}

/// Getters and setters
impl System {
    /// The current statistics of the simulation.
    pub fn stats(&self) -> &SystemStats {
        &self.stats
    }

    /// The current thermalisation ratio threshold.
    /// 
    /// See [`SystemBuilder::th_threshold`] for more information.
    pub fn th_threshold(&self) -> f64 {
        self.burn_in_desc.desired_ratio
    }

    pub fn current_step_size(&self) -> f64 {
        self.current_step_size.load(Ordering::Relaxed)
    }

    /// The current thermalisation block size.
    /// 
    /// See [`SystemBuilder::th_block_size`](SystemBuilder::th_block_size) for more information.
    pub fn th_block_size(&self) -> usize {
        self.burn_in_desc.avg_block_size
    }

    pub fn current_sweep(&self) -> usize {
        self.current_sweep
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
}