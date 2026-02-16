use crate::setup::{
    InitialState, LatticeCreateDesc, LatticeDesc, LatticeLoadDesc, SnapshotLocation,
};
use anyhow::Context;
use hdf5_metno as hdf5;
use ndarray::{Array4, Array5, ArrayView4, ArrayView5};
use num_traits::Pow;
use rand::{Rng, RngExt};
use std::cell::UnsafeCell;
use std::ops::Index;
use std::ops::Range;
use std::str::FromStr;

/// 2 adjacent indices in each dimension.
type AdjacentIndices = [usize; 8];

pub struct Lattice {
    pub(crate) sites: Vec<UnsafeCell<f64>>,
    spacing: f64,
    dimensions: [usize; 4],
    adjacency: Vec<AdjacentIndices>,
}

unsafe impl Send for Lattice {}
unsafe impl Sync for Lattice {}

impl Lattice {
    pub fn new(desc: LatticeCreateDesc) -> Self {
        match desc.initial_state {
            InitialState::Fixed(val) => Lattice::filled(desc.dimensions, desc.spacing, val),
            InitialState::RandomRange(range) => {
                Lattice::random(desc.dimensions, desc.spacing, range)
            }
        }
    }

    pub fn from_snapshot(desc: LatticeLoadDesc) -> anyhow::Result<Self> {
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

        let lattice = Lattice::from_view(raw_sites, desc.spacing);
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

    pub fn from_view(view: ArrayView4<f64>, spacing: f64) -> Self {
        let dimensions: (usize, usize, usize, usize) = view.dim();
        let dimensions: [usize; 4] = dimensions.into();

        let slice = view.as_slice().unwrap();
        let sites = slice
            .iter()
            .map(|&s| UnsafeCell::new(s))
            .collect::<Vec<_>>();

        let mut lattice = Lattice {
            sites,
            adjacency: Vec::new(),
            dimensions,
            spacing,
        };

        lattice.generate_adjacency();
        lattice
    }

    pub unsafe fn clone(&self) -> Self {
        let orig = &self.sites;
        let mut cloned = Vec::with_capacity(orig.len());

        unsafe {
            std::ptr::copy_nonoverlapping(orig.as_ptr(), cloned.as_mut_ptr(), orig.len());
            cloned.set_len(orig.len());
        }

        Self {
            sites: cloned,
            dimensions: self.dimensions,
            spacing: self.spacing,
            adjacency: self.adjacency.clone(),
        }
    }

    /// Computes the amount of iterations that a single sweep consists of.
    pub fn sweep_size(&self) -> usize {
        self.dimensions.iter().product()
    }

    /// Returns odd and even indices.
    pub fn generate_checkerboard(&self) -> (Vec<usize>, Vec<usize>) {
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

    /// Generates an adjacency table for the specified lattice.
    fn generate_adjacency(&mut self) -> Vec<AdjacentIndices> {
        let total_indices = self.sweep_size();
        let mut table = Vec::with_capacity(total_indices);

        for i in 0..total_indices {
            let mut neighbors = [0; 8];

            for j in 0..4 {
                let fneigh = self.get_forward_neighbor(i, j);
                let fneigh_index = self.to_index(fneigh);
                neighbors[2 * j] = fneigh_index;

                let bneigh = self.get_backward_neighbor(i, j);
                let bneigh_index = self.to_index(bneigh);
                neighbors[2 * j + 1] = bneigh_index;
            }

            table.push(neighbors);
        }

        table
    }

    pub fn spacing(&self) -> f64 {
        self.spacing
    }

    /// The dimensions of the lattice
    pub fn dimensions(&self) -> [usize; 4] {
        self.dimensions
    }

    /// The size of the lattice in the `t` dimension.
    pub fn dim_t(&self) -> usize {
        self.dimensions[0]
    }

    /// The size of the lattice in the `x` dimension.
    pub fn dim_x(&self) -> usize {
        self.dimensions[1]
    }

    /// The size of the lattice in the `y` dimension.
    pub fn dim_y(&self) -> usize {
        self.dimensions[2]
    }

    /// The size of the lattice in the `z` dimension.
    pub fn dim_z(&self) -> usize {
        self.dimensions[3]
    }

    /// Computes the mean of the lattice
    pub fn mean(&self) -> f64 {
        let sum: f64 = self.sites.iter().map(|c| unsafe { *c.get() }).sum();
        sum / self.sites.len() as f64
    }

    /// Computes the variance of the lattice
    pub fn meansq(&self) -> f64 {
        let sum: f64 = self.sites.iter().map(|x| unsafe { *x.get() }.pow(2)).sum();
        sum / self.sites.len() as f64
    }

    /// Computes the variance of the lattice.
    pub fn variance(&self) -> f64 {
        self.meansq() - self.mean().pow(2)
    }

    pub fn filled(dimensions: [usize; 4], spacing: f64, fill_value: f64) -> Self {
        let [t, x, y, z] = dimensions;
        let mut sites = Vec::with_capacity(t * x * y * z);

        for _ in 0..(t * x * y * z) {
            sites.push(UnsafeCell::new(fill_value));
        }

        let mut lattice = Self {
            sites,
            spacing,
            dimensions,
            adjacency: Vec::new(),
        };
        lattice.generate_adjacency();

        lattice
    }

    pub fn zeroed(dimensions: [usize; 4], spacing: f64) -> Self {
        let [t, x, y, z] = dimensions;
        tracing::debug!("Generating zeroed scalar lattice of dimensions {t} x {x} x {y} x {z}...");

        let mut sites = Vec::with_capacity(t * x * y * z);
        for _ in 0..(t * x * y * z) {
            sites.push(UnsafeCell::new(0.0));
        }

        tracing::debug!("Generated zeroed scalar lattice");

        let mut lattice = Self {
            sites,
            spacing,
            dimensions,
            adjacency: Vec::new(),
        };
        lattice.generate_adjacency();

        lattice
    }

    pub fn random(dimensions: [usize; 4], spacing: f64, range: Range<f64>) -> Self {
        let [t, x, y, z] = dimensions;
        tracing::debug!("Generating random scalar lattice of dimensions {t} x {x} x {y} x {z}...");

        let total_size = dimensions.iter().product();
        let mut sites = Vec::with_capacity(total_size);
        let mut rng = rand::rng();

        for _ in 0..total_size {
            sites.push(UnsafeCell::new(rng.random_range(range.clone())));
        }

        tracing::debug!("Generated random scalar lattice");

        let mut lattice = Self {
            sites,
            spacing,
            dimensions,
            adjacency: Vec::new(),
        };
        lattice.generate_adjacency();

        lattice
    }

    /// Gets the neighbor in the given forward direction. This implements wrapping of the boundaries.
    pub fn get_forward_neighbor(&self, orig: usize, dir: usize) -> [usize; 4] {
        // if !self.adjacency.is_empty() {
        //     return self.from_index(self.adjacency[orig][dir * 2])
        // }

        let mut dir_vec = [0; 4];
        dir_vec[dir] = 1;

        self.get_relative(self.from_index(orig), dir_vec)
    }

    /// Gets the neighbor in the given backward direction. This implements wrapping of the boundaries.
    pub fn get_backward_neighbor(&self, orig: usize, dir: usize) -> [usize; 4] {
        // if !self.adjacency.is_empty() {
        //     return self.from_index(self.adjacency[orig][dir * 2 + 1])
        // }

        let mut dir_vec = [0; 4];
        dir_vec[dir] = -1;

        self.get_relative(self.from_index(orig), dir_vec)
    }

    /// Gets the coordinates of a site relative to the current one in the given direction. This is necessary to introduce
    /// wrapping at the boundaries of the lattice.
    pub fn get_relative(&self, orig: [usize; 4], dir: [isize; 4]) -> [usize; 4] {
        let mut neighbor = [0; 4];
        for i in 0..4 {
            let ni = (orig[i] as isize + dir[i]).rem_euclid(self.dimensions[i] as isize) as usize;
            neighbor[i] = ni;
        }

        neighbor
    }

    /// Converts a lattice coordinate to an index.
    /// Periodic boundary conditions are imposed, i.e. the coordinates wrap around.
    pub fn to_index(&self, [t, x, y, z]: [usize; 4]) -> usize {
        let [st, sx, sy, sz] = self.dimensions;

        debug_assert!(t < st, "t coordinate out of range: {t} > {st}");
        debug_assert!(x < sx, "x coordinate out of range: {t} > {sx}");
        debug_assert!(y < sy, "y coordinate out of range: {t} > {sy}");
        debug_assert!(z < sz, "z coordinate out of range: {t} > {sz}");

        (t * sx * sy * sz) + (x * sy * sz) + (y * sz) + z
    }

    /// Converts a lattice index to a coordinate
    pub fn from_index(&self, i: usize) -> [usize; 4] {
        let [_, sx, sy, sz] = self.dimensions;
        let z = i % sz;

        let rem = (i - z) / sz;
        let y = rem % sy;

        let rem = (rem - y) / sy;
        let x = rem % sx;

        let t = (rem - x) / sx;

        [t, x, y, z]
    }
}

impl Index<usize> for Lattice {
    type Output = UnsafeCell<f64>;

    fn index(&self, i: usize) -> &Self::Output {
        &self.sites[i]
    }
}

impl Index<[usize; 4]> for Lattice {
    type Output = UnsafeCell<f64>;

    fn index(&self, pos: [usize; 4]) -> &Self::Output {
        &self.sites[self.to_index(pos)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test whether coordinates are mapped to indices into the vector storage correctly.
    #[test]
    fn lattice_index_map_test() {
        let dimensions = [5, 7, 13, 22];
        let lattice = Lattice::zeroed(dimensions, 1.0);

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
        let lattice = Lattice::zeroed(dimensions, 1.0);

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
