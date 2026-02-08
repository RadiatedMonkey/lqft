use std::ops::{Index, IndexMut};
use std::ops::Range;
use num_traits::Pow;
use rand::Rng;

pub struct ScalarLattice4D {
    sites: Vec<f64>,
    sizes: [usize; 4]
}

impl ScalarLattice4D {
    pub fn sweep_size(&self) -> usize {
        self.sizes.iter().product()
    }

    pub fn sizes(&self) -> [usize; 4] {
        self.sizes
    }

    /// Computes the mean of the lattice
    pub fn mean(&self) -> f64 {
        let sum: f64 = self.sites.iter().sum();
        sum / self.sites.len() as f64
    }
    
    /// Computes the variance of the lattice
    pub fn variance(&self) -> f64 {
        let sum: f64 = self.sites.iter().map(|x| x.pow(2)).sum();
        sum / self.sites.len() as f64
    }

    pub fn filled(sizes: [usize; 4], fill_value: f64) -> Self {
        let [t, x, y, z] = sizes;
        let mut sites = Vec::new();
        sites.resize(t * x * y * z, fill_value);

        Self { sites, sizes }
    }

    pub fn zeroed(sizes: [usize; 4]) -> Self {
        let [t, x, y, z] = sizes;
        println!("Generating zeroed scalar lattice of dimensions {t} x {x} x {y} x {z}");

        let mut sites = Vec::new();
        sites.resize(sizes.iter().product(), 0.0);

        println!("Generated zeroed scalar lattice");

        Self { sites, sizes }
    }

    pub fn random(sizes: [usize; 4], range: Range<f64>) -> Self {
        let [t, x, y, z] = sizes;
        println!("Generating random scalar lattice of dimensions {t} x {x} x {y} x {z}");

        let total_size = sizes.iter().product();
        let mut sites = Vec::with_capacity(total_size);
        let mut rng = rand::rng();
        for _ in 0..total_size {
            sites.push(rng.random_range(range.clone()));
        }

        println!("Generated random scalar lattice");

        Self { sites, sizes }
    }

    pub fn into_inner(self) -> Vec<f64> {
        self.sites
    }

    /// Gets the neighbor in the given forward direction. This implements wrapping of the boundaries.
    pub fn get_forward_neighbor(&self, orig: [usize; 4], dir: usize) -> [usize; 4] {
        let mut dir_vec = [0; 4];
        dir_vec[dir] = 1;

        self.get_relative(orig, dir_vec)
    }

    /// Gets the neighbor in the given backward direction. This implements wrapping of the boundaries.
    pub fn get_backward_neighbor(&self, orig: [usize; 4], dir: usize) -> [usize; 4] {
        let mut dir_vec = [0; 4];
        dir_vec[dir] = -1;

        self.get_relative(orig, dir_vec)
    }

    /// Gets the coordinates of a site relative to the current one in the given direction. This is necessary to introduce
    /// wrapping at the boundaries of the lattice.
    pub fn get_relative(&self, orig: [usize; 4], dir: [isize; 4]) -> [usize; 4] {
        let mut neighbor = [0; 4];
        for i in 0..4 {
            let ni = (orig[i] as isize + dir[i]).rem_euclid(self.sizes[i] as isize) as usize;
            neighbor[i] = ni;
        }

        neighbor
    }

    /// Converts a lattice coordinate to an index.
    /// Periodic boundary conditions are imposed, i.e. the coordinates wrap around.
    pub fn to_index(&self, [t, x, y, z]: [usize; 4]) -> usize {
        let [st, sx, sy, sz] = self.sizes;

        debug_assert!(t < st, "t coordinate out of range: {t} > {st}");
        debug_assert!(x < sx, "x coordinate out of range: {t} > {sx}");
        debug_assert!(y < sy, "y coordinate out of range: {t} > {sy}");
        debug_assert!(z < sz, "z coordinate out of range: {t} > {sz}");

        (t * sx * sy * sz) + (x * sy * sz) + (y * sz) + z
    }

    /// Converts a lattice index to a coordinate
    pub fn from_index(&self, i: usize) -> [usize; 4] {
        let [_, sx, sy, sz] = self.sizes;
        let z = i % sz;

        let rem = (i - z) / sz;
        let y = rem % sy;

        let rem = (rem - y) / sy;
        let x = rem % sx;

        let t = (rem - x) / sx;

        [t, x, y, z]
    }
}

impl Index<usize> for ScalarLattice4D {
    type Output = f64;

    fn index(&self, i: usize) -> &Self::Output {
        &self.sites[i]
    }
}

impl Index<[usize; 4]> for ScalarLattice4D {
    type Output = f64;

    fn index(&self, pos: [usize; 4]) -> &Self::Output {
        &self.sites[self.to_index(pos)]
    }
}

impl IndexMut<usize> for ScalarLattice4D {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.sites[i]
    }
}

impl IndexMut<[usize; 4]> for ScalarLattice4D {
    fn index_mut(&mut self, pos: [usize; 4]) -> &mut Self::Output {
        let idx = self.to_index(pos);
        &mut self.sites[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test whether coordinates are mapped to indices into the vector storage correctly.
    #[test]
    fn lattice_index_map_test() {
        let sizes = [5, 7, 13, 22];
        let lattice = ScalarLattice4D::zeroed(sizes);

        for i in 0..sizes.iter().product() {
            let coords = lattice.from_index(i);
            let idx = lattice.to_index(coords);

            println!("{coords:?}");
            assert_eq!(i, idx, "Conversion between indices and coordinates is incorrect!");
        }
    }

    /// Test whether exceeding boundaries correctly wraps back to the other side of the lattice.
    #[test]
    fn lattice_boundary_wrap_test() {
        let sizes = [5, 7, 13, 22];
        let lattice = ScalarLattice4D::zeroed(sizes);

        for (i, v) in sizes.iter().enumerate() {
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