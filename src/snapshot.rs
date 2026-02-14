use std::sync::{mpsc, Arc};
use std::thread;
use std::thread::JoinHandle;
use std::time::Instant;
use hdf5_metno as hdf5;
use ndarray::{Array5, ArrayView4, ArrayView5, Axis};
use crate::lattice::ScalarLattice4D;
use crate::stats::SweepStats;

/// Which method to use for flushing snapshot fragments.
#[derive(PartialEq, Copy, Clone)]
pub enum FlushMethod {
    /// Saves all fragments sequentially. This uses less RAM but is much slower.
    Sequential,
    /// Creates a batch of multiple sweeps and flushes them all at once.
    Batched
}

pub struct SnapshotFragment {
    pub(crate) lattice: ScalarLattice4D,
    pub(crate) stats: SweepStats
}

pub struct SnapshotState {
    pub flush_method: FlushMethod,
    pub file: hdf5::File,
    pub dataset: Arc<hdf5::Dataset>,
    pub chunk_size: [usize; 5],
    pub batch_size: usize,
    pub thread: Option<JoinHandle<()>>,
    pub tx: Option<mpsc::Sender<SnapshotFragment>>,
}

struct JobDesc {
    dataset: Arc<hdf5::Dataset>,
    rx: mpsc::Receiver<SnapshotFragment>,
    st: usize, sx: usize, sy: usize, sz: usize,
    batch_size: usize
}

impl SnapshotState {
    pub fn send_fragment(&self, fragment: SnapshotFragment) -> anyhow::Result<()> {
        let tx = self.tx.as_ref().ok_or_else(|| anyhow::anyhow!("Sender does not exist"))?;
        tx.send(fragment)?;

        Ok(())
    }

    fn sequential_job(desc: JobDesc) {
        let JobDesc {
            dataset, rx, st, sx, sy, sz, batch_size
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
            if let Err(err) = result { eprintln!("Failed to write snapshot fragment to disk: {err}") }

            println!("Flushed snapshot fragment {fragment_counter}");

            fragment_counter += 1;
        }
    }

    fn batch_job(desc: JobDesc) {
        let JobDesc {
            batch_size, dataset,
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
                    eprintln!("Failed to write snapshot batch: {err}");
                }

                println!("Wrote {batch_counter} snapshot batch to disk");

                fragment_counter += batch_size;
                batch_counter = 0;
            }
        }
    }

    pub fn init(&mut self, [st, sx, sy, sz]: [usize; 4], sweeps: usize) -> anyhow::Result<()> {
        if self.thread.is_some() {
            anyhow::bail!("Another simulation is already running");
        }

        self.dataset.resize([sweeps, st, sx, sy, sz])?;

        let (tx, rx) = mpsc::channel();
        self.tx = Some(tx);

        let job_desc = JobDesc {
            dataset: Arc::clone(&self.dataset), rx, st, sx, sy, sz, batch_size: self.batch_size
        };

        let flush_mode = self.flush_method;
        self.thread = Some(thread::spawn(move || {
            if flush_mode == FlushMethod::Sequential {
                Self::sequential_job(job_desc);
            } else {
                Self::batch_job(job_desc);
            }

            println!("Snapshot thread exited");
        }));

        println!("Snapshot state initialised");

        Ok(())
    }

    // pub fn update_snapshot(&mut self) -> anyhow::Result<()> {
    //     if let Some(set) = self.dataset.as_ref() {
    //         let lattice = &self.lattice.sites;
    //         let raw_slice: &[f64] = unsafe {
    //             std::slice::from_raw_parts(lattice.as_ptr() as *const f64, lattice.len())
    //         };
    //
    //         let [st, sx, sy, sz] = self.lattice.dimensions();
    //         let sweep_size = self.lattice.sweep_size();
    //
    //         assert_eq!(raw_slice.len(), sweep_size);
    //
    //         let view = ArrayView5::from_shape((1, st, sx, sy, sz), raw_slice)?;
    //         set.write_slice(
    //             view, ndarray::s![self.current_sweep..self.current_sweep+1, .., .., .., ..]
    //         )?;
    //     }
    //
    //     Ok(())
    // }
}

impl Drop for SnapshotState {
    fn drop(&mut self) {
        // Drop sender to signal to thread.
        let _ = self.tx.take();
        println!("Waiting for snapshot thread to shut down...");
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