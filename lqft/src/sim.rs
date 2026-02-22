use crate::lattice::Lattice;
use crate::setup::{AcceptanceDesc, BurnInDesc};
use crate::snapshot::{SnapshotState};
use crate::stats::SystemStats;
use atomic_float::AtomicF64;
use hdf5_metno::H5Type;
use num_traits::Pow;
use rand_xoshiro::rand_core::{Rng, SeedableRng};
use rand_xoshiro::{Xoshiro256PlusPlus, rand_core};
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use std::ops::{Div, Range};
use std::simd::num::SimdFloat;
use std::simd::{f64x4, u64x8, usizex4, Select, Simd, StdFloat};
use std::simd::prelude::SimdPartialOrd;
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool};
use std::time::Instant;
use rand::RngExt;
use crate::observable::{Observable, ObservableRegistry};
use crate::metrics::MetricState;
use crate::all_public_in;

const LANES: usize = 4;

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
    pub successful_therm_checks: usize
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

#[inline(always)]
fn to_coord(idx: usizex4, dims: [usize; 4]) -> [usizex4; 4] {
    let sx = usizex4::splat(dims[1]);
    let sy = usizex4::splat(dims[2]);
    let sz = usizex4::splat(dims[3]);

    let z = idx % sz;

    let rem = (idx - z) / sz;
    let y = rem % sy;

    let rem = (rem - y) / sy;
    let x = rem % sx;

    let t = (rem - x) / sx;

    [t, x, y, z]
}

#[inline(always)]
fn to_index(c: [usizex4; 4], dims: [usize; 4]) -> usizex4 {
    let sx = usizex4::splat(dims[1]);
    let sy = usizex4::splat(dims[2]);
    let sz = usizex4::splat(dims[3]);

    (c[0] * sx * sy * sz) + (c[1] * sy * sz) + (c[2] * sz) + c[3]
}

#[inline(always)]
fn find_fneighbors(site_idx: usizex4, dims: [usize; 4], dir: usize) -> usizex4 {
    let coords = to_coord(site_idx, dims);
    let mut new = coords;

    const ONE: usizex4 = usizex4::splat(1);
    let dim = usizex4::splat(dims[dir]);

    new[dir] = (coords[dir] + ONE) % dim;

    to_index(new, dims)
}

#[inline(always)]
fn find_bneighbors(site_idx: usizex4, dims: [usize; 4], dir: usize) -> usizex4 {
    let coords = to_coord(site_idx, dims);
    let mut new = coords;

    const ONE: usizex4 = usizex4::splat(1);
    let dim = usizex4::splat(dims[dir]);

    new[dir] = (coords[dir] + (dim - ONE)) % dim;

    to_index(new, dims)
}


impl System {
    /// Attempts to flip 8 sites starting from `idx`, returning the change in action.
    #[inline(always)]
    fn flip_site_chunk<C: Color, R: Rng>(
        rng: &mut R, idx: usize, chunk: &mut [f64], other_sites: &[f64], dims: [usize; 4], spacing: f64, step_size: f64, mass_squared: f64, coupling: f64
    ) -> u64 {
        let curr_vals = f64x4::from_slice(chunk);

        // `idx` is only the chunk index, convert the chunk index to 8 site indices.
        const CTR: usizex4 = usizex4::from_array([0, 1, 2, 3]);
        const TWO: usizex4 = usizex4::splat(2);

        let site_idxs = (usizex4::splat(idx) + CTR) * TWO;

        // Generate flip probabilities
        let mut offset = f64x4::splat(0.0);
        let mut realized = f64x4::splat(0.0);
        
        for i in 0..LANES {
            offset[i] = rng.random_range(-step_size..step_size);
            realized[i] = rng.random_range(0.0..1.0);
        }

        let new_vals = curr_vals + offset;

        let action_deltas = {
            let mut der_sum = f64x4::splat(0.0);
            let mut new_der_sum = f64x4::splat(0.0);

            let a = f64x4::splat(spacing);
            let div_a = f64x4::splat(1.0) / a;

            for i in 0..4 {
                let site_coords = Lattice::to_coord_multi(site_idxs, dims);
                let fneigh_vals = Lattice::load_forward_neighbor_multi(site_coords, i, other_sites, dims);
                let bneigh_vals = Lattice::load_backward_neighbor_multi(site_coords, i, other_sites, dims);

                // let fneigh_idxs = find_fneighbors(site_idxs, dimensions, i);
                // let bneigh_idxs = find_bneighbors(site_idxs, dimensions, i);

                // let fneigh_vals = f64x4::gather_or_default(neigh_sites, fneigh_idxs / TWO);
                // let bneigh_vals = f64x4::gather_or_default(neigh_sites, bneigh_idxs / TWO);

                {
                    let f1_sqrt = (fneigh_vals - curr_vals) * div_a;
                    let b1_sqrt = (curr_vals - bneigh_vals) * div_a;

                    let f1 = f1_sqrt * f1_sqrt;
                    let b1 = b1_sqrt * b1_sqrt;

                    der_sum += f1 + b1;
                }

                {
                    let f1_sqrt = (fneigh_vals - new_vals) * div_a;
                    let b1_sqrt = (bneigh_vals - new_vals) * div_a;

                    let f1 = f1_sqrt * f1_sqrt;
                    let b1 = b1_sqrt * b1_sqrt;

                    new_der_sum += f1 + b1;
                }
            }

            const HALF: f64x4 = f64x4::splat(0.5);
            const TFOURTH: f64x4 = f64x4::splat(1.0 / 24.0);;

            let msquared = f64x4::splat(mass_squared);
            let coupling = f64x4::splat(coupling);

            let curr_vals2 = curr_vals * curr_vals;
            let new_vals2 = new_vals * new_vals;

            let curr_vals4 = curr_vals2 * curr_vals2;
            let new_vals4 = new_vals2 * new_vals2;

            let kinetic_delta = HALF * (new_der_sum - der_sum);
            let mass_delta = HALF * msquared * (new_vals2 - curr_vals2);
            let inter_delta = TFOURTH * coupling * (new_vals4 - curr_vals4);

            let a4 = a * a * a * a;

            a4 * (kinetic_delta + mass_delta + inter_delta)
        };

        let prob_threshold = (-action_deltas).exp();
        let mask = realized.simd_lt(prob_threshold);
        let result = mask.select(new_vals, curr_vals);

        result.copy_to_slice(chunk);

        mask.to_array().iter().map(|&b| b as u64).sum()
    }

    /// Simulates the system using the checkerboard method.
    ///
    /// The checkerboard method works by dividing the lattice into a "checkerboard" of "red" and "black" sites.
    /// Since many QFTs interact with direct neighbours only, this allows all sites of a single colour to be updated
    /// simultaneously rather than sequentially.
    ///
    /// We first divide the lattice into a colours using [`generate_checkerboard`](Lattice::generate_checkerboard_indices).
    /// Then, using a parallel iterator, we iterate over every single lattice of a given colour and attempt to flip it
    /// using the Metropolis algorithm.
    pub fn simulate_checkerboard(&mut self, total_sweeps: usize) -> anyhow::Result<()> {
        self.data.stats.desired_sweeps = total_sweeps;

        let action = self.compute_full_action();
        self.data.stats.current_action = action;

        tracing::debug!("Starting action is: {action}");

        if let Some(state) = &mut self.snapshot_state {
            state.init(self.data.lattice.dimensions(), total_sweeps)?;
        }

        self.data.stats.reserve_capacity(total_sweeps);
        self.data.correlation_slices
            .resize(self.data.lattice.dimensions()[0], 0.0);

        tracing::info!("Running {total_sweeps} sweeps...");

        let mut sweep_timer;
        let mut total_timer = Instant::now();

        let mass_squared = self.data.mass_squared;
        let coupling = self.data.coupling;
        let spacing = self.data.lattice.spacing();
        let dimensions = self.data.lattice.dimensions();

        for i in 0..total_sweeps {
            sweep_timer = Instant::now();
            self.data.stats.current_sweep = i;

            // First update red sites....
            self.simulating.store(true, Ordering::SeqCst);

            let step_size = self.current_step_size();

            let action = &mut self.data.stats.current_action;
            let accepted_moves = &self.data.stats.accepted_moves;
            let total_moves = &self.data.stats.total_moves;

            let red_sites = &mut self.data.lattice.red_sites;
            let black_sites = &self.data.lattice.black_sites;

            red_sites.par_chunks_mut(LANES).enumerate().for_each_init(
                || {
                    let mut seed_rng = rand::rng();
                    Xoshiro256PlusPlus::from_rng(&mut seed_rng)
                },
                |rng, (j, chunk)| {
                    let new_accepted_moves = Self::flip_site_chunk::<Red, _>(
                        rng, j, chunk, black_sites,
                        dimensions,
                        spacing, step_size,
                        mass_squared, coupling
                    );

                    accepted_moves.fetch_add(new_accepted_moves, Ordering::SeqCst);
                    total_moves.fetch_add(8, Ordering::SeqCst);
                }
            );

            let red_sites = &self.data.lattice.red_sites;
            let black_sites = &mut self.data.lattice.black_sites;
            black_sites.par_chunks_mut(LANES).enumerate().for_each_init(
                || {
                    let mut seed_rng = rand::rng();
                    Xoshiro256PlusPlus::from_rng(&mut seed_rng)
                },
                |rng, (j, chunk)| {
                    let new_accepted_moves = Self::flip_site_chunk::<Black, _>(
                        rng, j, chunk, red_sites,
                        dimensions,
                        spacing, step_size,
                        mass_squared, coupling
                    );

                    accepted_moves.fetch_add(new_accepted_moves, Ordering::SeqCst);
                    total_moves.fetch_add(8, Ordering::SeqCst);
                }
            );

            self.simulating.store(false, Ordering::SeqCst);

            let sweep_time = sweep_timer.elapsed();
            sweep_timer = Instant::now();

            if i % 100 == 0 {
                // Recalculate total action
                let action = self.compute_full_action();
                self.data.stats.current_action = action;
            }

            // Keep track of thermalisation ratio.
            let (th_ratio, thermalised) = self.compute_burn_in_ratio();
            if i > 2 * self.data.burn_in_desc.block_size {
                self.data.stats.thermalisation_ratio_history.push(th_ratio);

                // Perform one check
                if i % self.data.burn_in_desc.block_size == 0 && self.data.stats.thermalised_at.is_none() {
                    if thermalised {
                        self.data.successful_therm_checks += 1;
                    } else {
                        self.data.successful_therm_checks = 0;
                    }

                    if self.data.successful_therm_checks == self.data.burn_in_desc.consecutive_passes {
                        // System has thermalised
                        self.data.stats.thermalised_at = Some(i);
                        tracing::info!("System has thermalised at sweep {i} after {} consecutive checks", self.data.burn_in_desc.consecutive_passes);
                    }
                }
            }

            if self.data.stats.thermalised_at.is_none() {
                self.correct_step_size();
            }

            // Record statistics on every sweep.
            self.record_stats(sweep_time, &sweep_timer, i, total_sweeps)?;

            if total_timer.elapsed().as_secs() >= 1 {
                self.push_metrics();
                total_timer = Instant::now();
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
        // TODO: Add sequential version

        let msquared = f64x4::splat(self.mass_squared());
        let coupling = f64x4::splat(self.coupling());
        let spacing = f64x4::splat(self.lattice().spacing());
        let a4 = spacing * spacing * spacing * spacing;
        let div_a = f64x4::splat(1.0) / spacing;
        let dims = self.lattice().dimensions();

        // Compute action for red sites

        let black_sites = &self.lattice().black_sites;
        let action_red = self
            .data.lattice.red_sites
            .par_chunks(8)
            .enumerate()
            .map(|(i, chunk)| {
                const CTR: usizex4 = usizex4::from_array([0, 1, 2, 3]);
                const TWO: usizex4 = usizex4::splat(2);

                let site_vals = f64x4::from_slice(chunk);
                let site_idxs = (usizex4::splat(i) + CTR) * TWO;

                let mut der_sum = f64x4::splat(0.0);
                for i in 0..4 {
                    let fneigh_idxs = find_fneighbors(site_idxs, dims, i);
                    // let bneigh_idxs = find_bneighbors(site_idxs, dims, i);

                    let fneigh_vals = f64x4::gather_or_default(black_sites, fneigh_idxs / TWO);
                    // let bneigh_vals = f64x4::gather_or_default(black_sites, bneigh_idxs / TWO);

                    let f1_sqrt = (fneigh_vals - site_vals) * div_a;
                    // let b1_sqrt = (site_vals - bneigh_vals) * div_a;

                    der_sum += f1_sqrt * f1_sqrt;
                }

                const HALF: f64x4 = f64x4::splat(0.5);
                const TFOURTH: f64x4 = f64x4::splat(1.0 / 24.0);

                let site_vals2 = site_vals * site_vals;
                let site_vals4 = site_vals2 * site_vals2;

                let kinetic = HALF * der_sum;
                let mass = HALF * msquared * site_vals2;
                let inter = TFOURTH * coupling * site_vals4;

                a4 * (kinetic + mass + inter)
            })
            .reduce(
                || f64x4::splat(0.0),
                |a, b| a + b
            )
            .reduce_sum();

        let red_sites = &self.lattice().red_sites;
        let action_black = self
            .data.lattice.black_sites
            .par_chunks(8)
            .enumerate()
            .map(|(i, chunk)| {
                const CTR: usizex4 = usizex4::from_array([0, 1, 2, 3]);
                const TWO: usizex4 = usizex4::splat(2);

                let site_vals = f64x4::from_slice(chunk);
                let site_idxs = (usizex4::splat(i) + CTR) * TWO;

                let mut der_sum = f64x4::splat(0.0);
                for i in 0..4 {
                    let fneigh_idxs = find_fneighbors(site_idxs, dims, i);
                    // let bneigh_idxs = find_bneighbors(site_idxs, dims, i);

                    let fneigh_vals = f64x4::gather_or_default(red_sites, fneigh_idxs / TWO);
                    // let bneigh_vals = f64x4::gather_or_default(red_sites, bneigh_idxs / TWO);

                    let f1_sqrt = (fneigh_vals - site_vals) * div_a;
                    // let b1_sqrt = (site_vals - bneigh_vals) * div_a;

                    der_sum += f1_sqrt * f1_sqrt
                }

                const HALF: f64x4 = f64x4::splat(0.5);
                const TFOURTH: f64x4 = f64x4::splat(1.0 / 24.0);

                let site_vals2 = site_vals * site_vals;
                let site_vals4 = site_vals2 * site_vals2;

                let kinetic = HALF * der_sum;
                let mass = HALF * msquared * site_vals2;
                let inter = TFOURTH * coupling * site_vals4;

                a4 * (kinetic + mass + inter)
            })
            .reduce(
                || f64x4::splat(0.0),
                |a, b| a + b
            )
            .reduce_sum();

        action_red + action_black
    }

    fn get_timeslice(&mut self) -> Vec<f64> {
        // Compute the spatial sum for every time slice
        let st = self.data.lattice.dim_t();

        let mut sum_t = vec![0.0; st];
        // for i in 0..self.data.lattice.sweep_size() {
        //     let t = self.data.lattice.from_index(i)[0];
        //     let val = unsafe { *self.data.lattice[i].get() };
        //     sum_t[t] += val;
        // }

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

pub enum Red {}
pub enum Black {}

pub trait Color {
    const IS_RED: bool;
}

impl Color for Red { const IS_RED: bool = true; }
impl Color for Black { const IS_RED: bool = false; }