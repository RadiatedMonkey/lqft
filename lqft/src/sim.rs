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
    /// We first divide the lattice into a colours using [`generate_checkerboard`](Lattice::generate_checkerboard).
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
        let (red, black) = self.data.lattice.generate_checkerboard();

        tracing::info!("Running {total_sweeps} sweeps...");

        let mut sweep_timer;
        let mut total_timer = Instant::now();

        for i in 0..total_sweeps {
            sweep_timer = Instant::now();
            self.data.stats.current_sweep = i;

            // First update red sites....
            self.simulating.store(true, Ordering::SeqCst);
            red.par_iter().for_each_init(
                || {
                    let mut seed_rng = rand::rng();
                    Xoshiro256PlusPlus::from_rng(&mut seed_rng)

                    // rand::rng()
                },
                |rng, &index| {
                    // SAFETY: Since this is a red site, this thread has exclusive access to the site.
                    // Therefore it can safely update the value.
                    unsafe {
                        let new_site = self.checkerboard_site_flip(rng, index);
                        *self.data.lattice[index].get() = new_site;
                    };

                    let index = index as u64;
                    if index % self.data.acceptance_desc.correction_interval == 0 && !self.thermalised()
                    {
                        self.correct_step_size();
                    }
                },
            );

            // then black sites.
            black.par_iter().for_each_init(
                || {
                    let mut seed_rng = rand::rng();
                    Xoshiro256PlusPlus::from_rng(&mut seed_rng)

                    // rand::rng()
                },
                |rng, &index| {
                    // SAFETY: Since this is a black site, this thread has exclusive access to the site.
                    // Therefore it can safely update the value.
                    unsafe {
                        let new_site = self.checkerboard_site_flip(rng, index);
                        *self.data.lattice[index].get() = new_site;
                    }

                    let index = index as u64;
                    if index % self.data.acceptance_desc.correction_interval == 0 && !self.thermalised()
                    {
                        self.correct_step_size();
                    }
                },
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

    fn random_range<R: rand_core::Rng>(rng: &mut R, range: Range<f64>) -> f64 {
        let rand_full = rng.next_u64() as f64;
        let rand_unit = rand_full / (u64::MAX as f64 + 1.0);

        range.start + (rand_unit * (range.end - range.start))
    }

    /// Performs a single iteration and returns the new field value at the given site.
    ///
    /// SAFETY: This method should only be called if the calling thread has exclusive access to the given site.
    /// and the direct neighbours can safely be read from.
    #[inline(always)]
    unsafe fn checkerboard_site_flip<R: rand_core::Rng>(&self, rng: &mut R, site: usize) -> f64 {
        let a = self.data.lattice.spacing();

        let curr_val = unsafe { *self.data.lattice[site].get() };

        let step_size = self.current_step_size();
        let new_val = Self::random_range(rng, (curr_val - step_size)..(curr_val + step_size));

        let mut curr_der_sum = 0.0;
        let mut new_der_sum = 0.0;

        let scurr_val = Simd::from([curr_val; 4]);
        let snew_val = Simd::from([new_val; 4]);

        let mut fneighbours = Simd::from([0.0; 4]);
        let mut bneighbours = Simd::from([0.0; 4]);

        for i in 0..4 {
            // let orig = self.lattice.from_index(site);
            let fneigh = self.data.lattice.get_forward_neighbor(site, i);
            let bneigh = self.data.lattice.get_backward_neighbor(site, i);

            // SAFETY: Neighbor sites will always only be read from due to checkerboarding.
            let fneigh_val = unsafe { *self.data.lattice[fneigh].get() };
            let bneigh_val = unsafe { *self.data.lattice[bneigh].get() };

            // curr_der_sum +=
            //     ((fneigh_val - curr_val) / a).pow(2) + ((curr_val - bneigh_val) / a).pow(2);
            // new_der_sum +=
            //     ((fneigh_val - new_val) / a).pow(2) + ((new_val - bneigh_val) / a).pow(2);

            fneighbours[i] = fneigh_val;
            bneighbours[i] = bneigh_val;
        }

        let curr_der_sum = {
            let fdiff = fneighbours - scurr_val;
            let fsquared = fdiff * fdiff;
            let bdiff = scurr_val - bneighbours;
            let bsquared = bdiff * bdiff;

            (fsquared + bsquared).reduce_sum()
        };

        let new_der_sum = {
            let fdiff = fneighbours - snew_val;
            let fsquared = fdiff * fdiff;
            let bdiff = snew_val - bneighbours;
            let bsquared = bdiff * bdiff;

            (fsquared + bsquared).reduce_sum()
        };

        let kinetic_delta = 0.5 * (new_der_sum - curr_der_sum);
        let mass_delta = 0.5 * self.data.mass_squared * (new_val * new_val - curr_val * curr_val);
        let interaction_delta = 1.0 / 24.0 * self.data.coupling * (new_val.pow(4) - curr_val.pow(4));

        let total_delta: f64 = a.pow(4) * (kinetic_delta + mass_delta + interaction_delta);
        let accept_prob = (-total_delta).exp();
        let realised = Self::random_range(rng, 0.0..1.0);

        self.data.stats.total_moves.fetch_add(1, Ordering::Relaxed);
        if realised < accept_prob {
            self.data
                .stats
                .current_action
                .fetch_add(total_delta, Ordering::AcqRel);

            self.data.stats.accepted_moves.fetch_add(1, Ordering::Relaxed);
            return new_val;
        }

        curr_val
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
