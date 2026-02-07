use std::ops::Index;
use rand::Rng;

pub struct Lattice4D<const N: usize> {
    sites: Vec<f64>
}

impl<const N: usize> Lattice4D<N> {
    const SIZE: usize = N.pow(4);

    pub const fn size(&self) -> usize {
        Self::SIZE
    }

    pub fn zeroed() -> Self {
        let mut sites = Vec::new();
        sites.resize(Self::SIZE, 0.0);

        Self { sites }
    }

    pub fn random() -> Self {
        println!("Generating random 4D lattice of size {N}^4");

        let mut sites = Vec::with_capacity(Self::SIZE);
        let mut rng = rand::rng();
        for _ in 0..Self::SIZE {
            sites.push(rng.random_range(-20.0..20.0));
        }

        println!("Generated random lattice");

        Self { sites }
    }

    fn impose_wrapping([t, x, y, z]: [usize; 4]) {

    }

    pub fn into_inner(self) -> Vec<f64> {
        self.sites
    }

    /// Converts a lattice coordinate to an index.
    /// Periodic boundary conditions are imposed, i.e. the coordinates wrap around.
    pub const fn to_index([t, x, y, z]: [usize; 4]) -> usize {
        t * N.pow(3) + x * N.pow(2) + y * N + z
    }

    /// Converts a lattice index to a coordinate
    pub const fn from_index(i: usize) -> [usize; 4] {
        let z = i % N;

        let rem = i - z;
        let yN = rem % N.pow(2);
        let y = yN / N;

        let rem = rem - yN;
        let xN2 = rem % N.pow(3);
        let x = xN2 / N.pow(2);

        let rem = rem - xN2;
        let t = rem / N.pow(3);

        [t, x, y, z]
    }
}

impl<const N: usize> Index<usize> for Lattice4D<N> {
    type Output = f64;

    fn index(&self, i: usize) -> &Self::Output {
        &self.sites[i]
    }
}

impl<const N: usize> Index<[usize; 4]> for Lattice4D<N> {
    type Output = f64;

    fn index(&self, pos: [usize; 4]) -> &Self::Output {
        &self.sites[Self::to_index(pos)]
    }
}

// use std::ops::{AddAssign, Div, Index, IndexMut};
// use num_traits::{ConstZero, Zero};
//
// /// Floating point type to use. Should be either f32 or f64.
// type Prec = f64;
//
// pub struct SiteLattice<const N: usize, const M: usize> {
//     data: [[Prec; N]; M]
// }
//
// impl<const N: usize, const M: usize> SiteLattice<N, M> {
//     pub const fn zeroed() -> Self {
//         Self { data: [[0.0; N]; M] }
//     }
//
//     /// Returns the total amount of lattice sites.
//     pub const fn size() -> usize {
//         N * M
//     }
// }
//
// impl<const N: usize, const M: usize> Index<(usize, usize)> for SiteLattice<N, M> {
//     type Output = Prec;
//
//     fn index(&self, index: (usize, usize)) -> &Self::Output {
//         &self.data[index.0][index.1]
//     }
// }
//
// impl<const N: usize, const M: usize> IndexMut<(usize, usize)> for SiteLattice<N, M> {
//     fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
//         &mut self.data[index.0][index.1]
//     }
// }
//
// impl<const N: usize, const M: usize> From<[[Prec; N]; M]> for SiteLattice<N, M> {
//     fn from(data: [[Prec; N]; M]) -> Self {
//         Self { data }
//     }
// }