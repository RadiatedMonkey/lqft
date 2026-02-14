use std::sync::{mpsc, Arc};
use std::thread;
use std::thread::JoinHandle;
use std::time::Instant;
use hdf5_metno as hdf5;
use ndarray::{Array5, ArrayView4, ArrayView5, Axis};
use crate::lattice::ScalarLattice4D;
use crate::setup::SnapshotDesc;
use crate::stats::SweepStats;

/// Which method to use for flushing snapshots fragments.
#[derive(PartialEq, Copy, Clone)]
pub enum FlushMethod {
    /// Saves all fragments sequentially. This uses less RAM but is much slower.
    Sequential,
    /// Creates a batch of `n` sweeps and flushes them all at once.
    Batched(usize)
}

/// A single snapshots fragment.
///
/// This represents one sweep together with its statistics.
pub struct SnapshotFragment {
    pub(crate) lattice: ScalarLattice4D,
    pub(crate) stats: SweepStats
}

pub struct SnapshotState {
    pub file: hdf5::File,
    pub dataset: Arc<hdf5::Dataset>,
    pub desc: SnapshotDesc,
    pub thread: Option<JoinHandle<()>>,
    pub tx: Option<mpsc::Sender<SnapshotFragment>>,
}

struct JobDesc {
    dataset: Arc<hdf5::Dataset>,
    rx: mpsc::Receiver<SnapshotFragment>,
    st: usize, sx: usize, sy: usize, sz: usize
}

impl SnapshotState {
    pub fn send_fragment(&self, fragment: SnapshotFragment) -> anyhow::Result<()> {
        let tx = self.tx.as_ref().ok_or_else(|| anyhow::anyhow!("Sender does not exist"))?;
        tx.send(fragment)?;

        Ok(())
    }

    fn sequential_job(desc: JobDesc) {
        let JobDesc {
            dataset,
            rx, st, sx, sy, sz
        } = desc;

        let mut fragment_counter = 0;

        while let Ok(fragment) = rx.recv() {
            let sites = &fragment.lattice.sites;
            let raw_slice: &[f64] = unsafe {
                std::slice::from_raw_parts(sites.as_ptr() as *const f64, sites.len())
            };

            let view = ArrayView5::from_shape(
                (1, st, sx, sy, sz), raw_slice
            );

            if let Err(err) = &view { eprintln!("Failed to create ArrayView5: {err}") }
            let view = view.unwrap();

            let result = dataset.write_slice(
                view, ndarray::s![fragment_counter..fragment_counter + 1, .., .., .., ..]
            );
            if let Err(err) = result { eprintln!("Failed to write snapshots fragment to disk: {err}") }

            println!("Flushed snapshots fragment {fragment_counter}");

            fragment_counter += 1;
        }
    }

    fn batch_job(desc: JobDesc, batch_size: usize) {
        let JobDesc {
            dataset,
            rx,
            st, sx, sy, sz,
        } = desc;

        let mut buffer = Array5::<f64>::zeros((batch_size, st, sx, sy, sz));

        let mut fragment_counter = 0;
        let mut batch_counter = 0;
        while let Ok(fragment) = rx.recv() {
            let sites = &fragment.lattice.sites;
            let raw_slice: &[f64] = unsafe {
                std::slice::from_raw_parts(sites.as_ptr() as *const f64, sites.len())
            };

            let mut subview = buffer.slice_mut(
                ndarray::s![batch_counter, .., .., .., ..]
            );
            let incoming_view = ArrayView4::from_shape(
                (st, sx, sy, sz), raw_slice
            ).unwrap();
            subview.assign(&incoming_view);

            batch_counter += 1;

            if batch_counter >= batch_size {
                let selection = ndarray::s![
                    fragment_counter..fragment_counter + batch_counter, .., .., .., ..
                ];

                if let Err(err) = dataset.write_slice(&buffer, selection) {
                    eprintln!("Failed to write snapshots batch: {err}");
                }

                println!("Wrote {batch_counter} snapshots batch to disk");

                fragment_counter += batch_size;
                batch_counter = 0;
            }
        }
    }

    /// Initialises the snapshots.
    pub fn init(&mut self, [st, sx, sy, sz]: [usize; 4], sweeps: usize) -> anyhow::Result<()> {
        if self.thread.is_some() {
            anyhow::bail!("Another simulation is already running");
        }

        self.dataset.resize([sweeps, st, sx, sy, sz])?;

        let (tx, rx) = mpsc::channel();
        self.tx = Some(tx);

        let job_desc = JobDesc {
            dataset: Arc::clone(&self.dataset),
            rx, st, sx, sy, sz
        };

        let method = self.desc.flush_method;
        self.thread = Some(thread::spawn(move || {
            match method {
                FlushMethod::Sequential => Self::sequential_job(job_desc),
                FlushMethod::Batched(size) => Self::batch_job(job_desc, size)
            }

            println!("Snapshot thread exited");
        }));

        println!("Snapshot state initialised");

        Ok(())
    }

}

impl Drop for SnapshotState {
    fn drop(&mut self) {
        // Drop sender to signal to thread.
        let _ = self.tx.take();
        println!("Waiting for snapshots thread to shut down...");
        let now = Instant::now();
        let _ = self.thread.take().map(JoinHandle::join);
        let elapsed = now.elapsed().as_millis();
        println!("Time elapsed: {elapsed}");
    }
}

/// Creates the HDF5 file and prepares it for storage.
pub fn create_hdf5_file(filename: &str) -> anyhow::Result<hdf5::File> {
    let file = hdf5::File::create(filename)?;

    Ok(file)
}