use crate::setup::{InitialState, LatticeCreateDesc, LatticeDesc, LatticeIterMethod, LatticeLoadDesc, SnapshotLocation};
use anyhow::Context;
use hdf5_metno as hdf5;
use ndarray::{Array4, Array5, ArrayView4, ArrayView5};
use num_traits::Pow;
use rand::{Rng, RngExt};
use std::cell::UnsafeCell;
use std::ops::{Deref, Index};
use std::ops::Range;
use std::str::FromStr;

use rayon::prelude::*;

/// 2 adjacent indices in each dimension.
type AdjacentIndices = [usize; 8];

#[derive(Clone)]
pub struct Lattice<const Dim: usize> {
    iter_method: LatticeIterMethod,
    spacing: f64,
    dimensions: [usize; Dim],

    pub(crate) red_sites: Vec<f64>,
    pub(crate) black_sites: Vec<f64>
}

unsafe impl<const Dim: usize> Send for Lattice<Dim> {}
unsafe impl<const Dim: usize> Sync for Lattice<Dim> {}

impl<const Dim: usize> Lattice<Dim> {
    pub fn new(desc: LatticeCreateDesc<Dim>, iter_method: LatticeIterMethod) -> Self {
        match desc.initial_state {
            InitialState::Fixed(val) => Lattice::filled(desc.dimensions, desc.spacing, iter_method, val),
            InitialState::RandomRange(range) => {
                Lattice::random(desc.dimensions, desc.spacing, iter_method, range)
            }
        }
    }

    pub fn from_snapshot(desc: LatticeLoadDesc, iter_method: LatticeIterMethod) -> anyhow::Result<Self> {
        let file = hdf5::File::open(desc.hdf5_file).context("Unable to open snapshots file")?;

        let set = match desc.location {
            SnapshotLocation::Direct(loc) => file.dataset(&loc)?,
            SnapshotLocation::Latest(group_name) => {
                let group = file
                    .group(&group_name)
                    .context("Unable to find snapshot group")?;
                let sets = group.datasets()?;

                if sets.is_empty() {
                    anyhow::bail!(
                        "Unable to find latest snapshot, HDF5 group {group_name} is empty"
                    )
                }

                let set = sets
                    .iter()
                    .max_by_key(|set| {
                        let full_name = &set.name();
                        if let Some(name) = full_name.split("/").last() {
                            let res = u64::from_str(name);
                            res.unwrap_or_else(|_| {
                                tracing::warn!(
                                    "Unable to parse snapshot name \"{name}\", ignoring it"
                                );
                                0
                            })
                        } else {
                            tracing::warn!(
                                "Unable to parse snapshot name \"{full_name}\", ignoring it"
                            );
                            0
                        }
                    })
                    .unwrap();

                tracing::info!("Loading dataset \"{}\"...", set.name());

                set.clone()
            }
        };

        let set_name = set.name();

        let set_shape = set.shape();
        let ndim = set_shape.len();
        if ndim != 5 {
            anyhow::bail!(
                "Unable to load snapshot \"{set_name}\", dataspace has {ndim} != 5 dimensions"
            )
        }

        let latest_idx = set_shape[0] - 1;
        let selection = ndarray::s![latest_idx..latest_idx + 1, .., .., .., ..];

        let raw_slice: Array5<f64> = set
            .read_slice(selection)
            .context("Unable to read dataset")?;
        let raw_sites: ArrayView4<f64> = raw_slice.slice(ndarray::s![0, .., .., .., ..]);

        let lattice = Lattice::from_view(raw_sites, desc.spacing, iter_method);
        tracing::info!(
            "Loaded lattice of dimensions {} x {} x {} x {} from file \"{}\"",
            set_shape[1],
            set_shape[2],
            set_shape[3],
            set_shape[4],
            set_name
        );

        file.close()?;

        Ok(lattice)
    }

    pub fn iter_method(&self) -> LatticeIterMethod {
        self.iter_method
    }

    pub fn from_view(view: ArrayView4<f64>, spacing: f64, iter_method: LatticeIterMethod) -> Self {
        let dimensions: (usize, usize, usize, usize) = view.dim();

        let count = view.len().div_ceil(2);
        let mut red_sites = vec![0.0; count];
        let mut black_sites = vec![0.0; count];

        // for t in 0..dimensions.0 {
        //     for x in 0..dimensions.1 {
        //         for y in 0..dimensions.2 {
        //             for z in 0..dimensions.3 {
        //                 if (t + x + y + z) % 2 == 0 {
        //                     // Red site
        //
        //                 } else {
        //                     // Black site
        //                 }
        //             }
        //         }
        //     }
        // }
        todo!();

        // let mut lattice = Lattice {
        //     iter_method,
        //     red_sites: UnsafeCell::new(red_sites), black_sites: UnsafeCell::new(black_sites),
        //     adjacency: Vec::new(),
        //     dimensions: dimensions.into(),
        //     spacing,
        // };
        //
        // lattice.generate_adjacency();
        // lattice
    }

    /// Computes the amount of iterations that a single sweep consists of.
    pub fn sweep_size(&self) -> usize {
        self.dimensions.iter().product()
    }

    /// Returns odd and even indices.
    pub fn generate_checkerboard_indices(&self) -> (Vec<usize>, Vec<usize>) {
        let [st, sx, sy, sz] = self.dimensions;
        let stotal: usize = self.dimensions.iter().sum();

        let mut red = Vec::with_capacity(stotal / 2 + 1);
        let mut black = Vec::with_capacity(stotal / 2 + 1);

        tracing::debug!("Generating checkerboard indices...");

        // Generate red indices
        for t in 0..st {
            for x in 0..sx {
                for y in 0..sy {
                    for z in 0..sz {
                        let index = self.to_index([t, x, y, z]);
                        if (t + x + y + z) % 2 == 0 {
                            red.push(index);
                        } else {
                            black.push(index);
                        }
                    }
                }
            }
        }

        tracing::debug!("Checkerboard generated!");

        (red, black)
    }

    pub fn spacing(&self) -> f64 {
        self.spacing
    }

    /// The dimensions of the lattice
    pub fn dimensions(&self) -> [usize; Dim] {
        self.dimensions
    }

    pub fn mean(&self) -> f64 {
        match self.iter_method {
            LatticeIterMethod::Sequential => self.mean_seq(),
            LatticeIterMethod::Parallel => self.mean_par()
        }
    }

    /// Sequentially computes the mean of the lattice
    #[inline]
    pub fn mean_seq(&self) -> f64 {
        let red_sum = self.red_sites.iter().sum::<f64>();
        let black_sum = self.black_sites.iter().sum::<f64>();

        (red_sum + black_sum) / self.sweep_size() as f64
    }

    /// Computes the mean of the lattice in parallel.
    #[inline]
    pub fn mean_par(&self) -> f64 {
        let red_sum = self.red_sites.par_iter().sum::<f64>();
        let black_sum = self.black_sites.par_iter().sum::<f64>();

        (red_sum + black_sum) / self.sweep_size() as f64
    }

    pub fn meansq(&self) -> f64 {
        match self.iter_method {
            LatticeIterMethod::Sequential => self.meansq_seq(),
            LatticeIterMethod::Parallel => self.meansq_par()
        }
    }

    /// Sequentially computes the mean squared of the lattice
    #[inline]
    pub fn meansq_seq(&self) -> f64 {
        let red_sum = self.red_sites.iter().map(|x| x * x).sum::<f64>();
        let black_sum = self.black_sites.iter().map(|x| x * x).sum::<f64>();

        (red_sum + black_sum) / self.sweep_size() as f64
    }

    #[inline]
    /// Computes the mean squared of the lattice in parallel
    pub fn meansq_par(&self) -> f64 {
        let red_sum = self.red_sites.par_iter().map(|x| x * x).sum::<f64>();
        let black_sum = self.black_sites.par_iter().map(|x| x * x).sum::<f64>();

        (red_sum + black_sum) / self.sweep_size() as f64
    }

    pub fn variance(&self) -> f64 {
        match self.iter_method {
            LatticeIterMethod::Sequential => self.variance_seq(),
            LatticeIterMethod::Parallel => self.variance_par()
        }
    }

    /// Sequentially computes the variance of the lattice.
    #[inline]
    pub fn variance_seq(&self) -> f64 {
        let mean = self.mean_seq();
        self.meansq_seq() - mean * mean
    }

    #[inline]
    /// Computes the variance of the lattice in parallel.
    pub fn variance_par(&self) -> f64 {
        let mean = self.mean_par();
        self.meansq_par() - mean * mean
    }

    /// Fills the data with a fixed value.
    pub fn filled(dims: [usize; Dim], spacing: f64, iter_method: LatticeIterMethod, fill_value: f64) -> Self {
        let count = dims.iter().product::<usize>().div_ceil(2);
        let red_sites = vec![fill_value; count];
        let black_sites = vec![fill_value; count];

        Self {
            iter_method,
            red_sites, black_sites,
            spacing,
            dimensions: dims
        }
    }

    /// Fills the data with zeroes.
    #[inline]
    pub fn zeroed(dims: [usize; Dim], spacing: f64, iter_method: LatticeIterMethod) -> Self {
        Self::filled(dims, spacing, iter_method, 0.0)
    }

    /// Fills the lattice with random data from a range.
    pub fn random(dims: [usize; Dim], spacing: f64, iter_method: LatticeIterMethod, range: Range<f64>) -> Self {
        tracing::debug!("Generating random scalar lattice of dimensions {dims:?}...");

        let count = dims.iter().product::<usize>().div_ceil(2);
        let mut rng = rand::rng();

        // Seems like I need to clone `range` because the map function can only capture it once.
        let red_sites = (0..count)
            .map(|_| rng.random_range(range.clone()))
            .collect::<Vec<_>>();

        let black_sites = (0..count)
            .map(|_| rng.random_range(range.clone()))
            .collect::<Vec<_>>();

        Self {
            iter_method,
            red_sites, black_sites,
            spacing,
            dimensions: dims
        }
    }

    /// Gets the neighbor in the given forward direction. This implements wrapping of the boundaries.
    pub fn get_forward_neighbor(&self, orig: usize, dir: usize) -> [usize; Dim] {
        // if !self.adjacency.is_empty() {
        //     return self.from_index(self.adjacency[orig][dir * 2])
        // }

        let mut dir_vec = [0; Dim];
        dir_vec[dir] = 1;

        self.get_relative(self.from_index(orig), dir_vec)
    }

    /// Gets the neighbor in the given backward direction. This implements wrapping of the boundaries.
    pub fn get_backward_neighbor(&self, orig: usize, dir: usize) -> [usize; Dim] {
        // if !self.adjacency.is_empty() {
        //     return self.from_index(self.adjacency[orig][dir * 2 + 1])
        // }

        let mut dir_vec = [0; Dim];
        dir_vec[dir] = -1;

        self.get_relative(self.from_index(orig), dir_vec)
    }

    /// Gets the coordinates of a site relative to the current one in the given direction. This is necessary to introduce
    /// wrapping at the boundaries of the lattice.
    pub fn get_relative(&self, orig: [usize; Dim], dir: [isize; Dim]) -> [usize; Dim] {
        let mut neighbor = [0; Dim];
        for i in 0..4 {
            let ni = (orig[i] as isize + dir[i]).rem_euclid(self.dimensions[i] as isize) as usize;
            neighbor[i] = ni;
        }

        neighbor
    }

    /// Converts a lattice coordinate to an index.
    /// Periodic boundary conditions are imposed, i.e. the coordinates wrap around.
    pub fn to_index(&self, coords: [usize; Dim]) -> usize {
        let mut mult = 1;
        let mut idx = 0;
        for d in Dim..0 {
            idx += coords[d] * mult;            
            mult *= self.dimensions[d];
        }

        idx

        // (t * sx * sy * sz) + (x * sy * sz) + (y * sz) + z
    }

    /// Converts a lattice index to a coordinate
    pub fn from_index(&self, i: usize) -> [usize; Dim] {
        let [_, sx, sy, sz] = self.dimensions;
        let z = i % sz;

        let rem = (i - z) / sz;
        let y = rem % sy;

        let rem = (rem - y) / sy;
        let x = rem % sx;

        let t = (rem - x) / sx;

        [t, x, y, z]
    }

    #[inline]
    fn is_red(&self, coord: [usize; Dim]) -> bool {
        coord.iter().sum::<usize>() % 2 == 0
    }
}

// impl Index<usize> for Lattice {
//     type Output = UnsafeCell<f64>;
//
//     fn index(&self, i: usize) -> &Self::Output {
//         &self.sites[i]
//     }
// }
//
// impl Index<[usize; 4]> for Lattice {
//     type Output = UnsafeCell<f64>;
//
//     fn index(&self, pos: [usize; 4]) -> &Self::Output {
//         &self.sites[self.to_index(pos)]
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    /// Test whether coordinates are mapped to indices into the vector storage correctly.
    #[test]
    fn lattice_index_map_test() {
        let dimensions = [5, 7, 13, 22];
        let lattice = Lattice::zeroed(dimensions, 1.0, LatticeIterMethod::Sequential);

        for i in 0..dimensions.iter().product() {
            let coords = lattice.from_index(i);
            let idx = lattice.to_index(coords);

            println!("{coords:?}");
            assert_eq!(
                i, idx,
                "Conversion between indices and coordinates is incorrect!"
            );
        }
    }

    /// Test whether exceeding boundaries correctly wraps back to the other side of the lattice.
    #[test]
    fn lattice_boundary_wrap_test() {
        let dimensions = [5, 7, 13, 22];
        let lattice = Lattice::zeroed(dimensions, 1.0, LatticeIterMethod::Sequential);

        for (i, v) in dimensions.iter().enumerate() {
            let mut start = [0; 4];
            start[i] = *v - 1;

            let mut dir = [0; 4];
            dir[i] = 1;

            let neighbor = lattice.get_relative(start, dir);
            println!("{start:?} + {dir:?} = {neighbor:?}");
            assert_eq!(neighbor, [0; 4]);

            dir[i] = -(start[i] as isize);
            let neighbor = lattice.get_relative(start, dir);

            println!("{start:?} + {dir:?} = {neighbor:?}");
            assert_eq!(neighbor, [0; 4]);

            println!("Coordinate {i} is correct");
        }
    }
}
