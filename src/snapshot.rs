use crate::lattice::Lattice;
use crate::setup::{FlushMethod, SnapshotDesc, SnapshotType};
use crate::stats::SweepStats;
use hdf5_metno as hdf5;
use ndarray::{Array5, ArrayView4, ArrayView5, Axis};
use std::ops::Div;
use std::sync::{Arc, mpsc};
use std::thread;
use std::thread::JoinHandle;
use std::time::Instant;

/// A single snapshots fragment.
///
/// This represents one sweep together with its statistics.
pub struct SnapshotFragment {
    pub(crate) lattice: Lattice,
    pub(crate) stats: SweepStats,
}

pub struct SnapshotState {
    pub file: Option<hdf5::File>,
    pub dataset: Arc<hdf5::Dataset>,
    pub desc: SnapshotDesc,
    pub thread: Option<JoinHandle<()>>,
    pub tx: Option<mpsc::Sender<SnapshotFragment>>,
}

struct JobDesc {
    dataset: Arc<hdf5::Dataset>,
    rx: mpsc::Receiver<SnapshotFragment>,
    st: usize,
    sx: usize,
    sy: usize,
    sz: usize,
}

impl SnapshotState {
    pub fn send_fragment(&self, fragment: SnapshotFragment) -> anyhow::Result<()> {
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Sender does not exist"))?;
        tx.send(fragment)?;

        Ok(())
    }

    fn sequential_job(desc: JobDesc) {
        tracing::info!("Initialised sequential snapshot flush thread");

        let JobDesc {
            dataset,
            rx,
            st,
            sx,
            sy,
            sz,
        } = desc;

        let mut fragment_counter = 0;

        while let Ok(fragment) = rx.recv() {
            tracing::trace!("Snapshot fragment {fragment_counter} received");

            let sites = &fragment.lattice.sites;
            let raw_slice: &[f64] =
                unsafe { std::slice::from_raw_parts(sites.as_ptr() as *const f64, sites.len()) };

            let view = ArrayView5::from_shape((1, st, sx, sy, sz), raw_slice);

            if let Err(err) = &view {
                tracing::error!("Failed to create ArrayView5: {err}")
            }
            let view = view.unwrap();

            let result = dataset.write_slice(
                view,
                ndarray::s![fragment_counter..fragment_counter + 1, .., .., .., ..],
            );
            if let Err(err) = result {
                tracing::error!("Failed to write snapshots fragment to disk: {err}")
            }

            tracing::trace!("Flushed snapshots fragment {fragment_counter}");

            fragment_counter += 1;
        }
    }

    fn batch_job(desc: JobDesc, batch_size: usize) {
        tracing::info!("Initialised snapshot batch flush thread");

        let JobDesc {
            dataset,
            rx,
            st,
            sx,
            sy,
            sz,
        } = desc;

        let mut buffer = Array5::<f64>::zeros((batch_size, st, sx, sy, sz));

        let mut fragment_counter = 0;
        let mut batch_counter = 0;
        while let Ok(fragment) = rx.recv() {
            let sites = &fragment.lattice.sites;
            let raw_slice: &[f64] =
                unsafe { std::slice::from_raw_parts(sites.as_ptr() as *const f64, sites.len()) };

            let mut subview = buffer.slice_mut(ndarray::s![batch_counter, .., .., .., ..]);
            let incoming_view = ArrayView4::from_shape((st, sx, sy, sz), raw_slice).unwrap();
            subview.assign(&incoming_view);

            batch_counter += 1;

            if batch_counter >= batch_size {
                let selection = ndarray::s![
                    fragment_counter..fragment_counter + batch_counter,
                    ..,
                    ..,
                    ..,
                    ..
                ];

                if let Err(err) = dataset.write_slice(&buffer, selection) {
                    tracing::error!("Failed to write snapshots batch: {err}");
                }

                tracing::trace!("Wrote {batch_counter} snapshots batch to disk");

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

        let snapshot_count = match self.desc.ty {
            SnapshotType::Interval(interval) => sweeps / interval,
            SnapshotType::Checkpoint => 1,
        };

        self.dataset.resize([snapshot_count, st, sx, sy, sz])?;

        let (tx, rx) = mpsc::channel();
        self.tx = Some(tx);

        let job_desc = JobDesc {
            dataset: Arc::clone(&self.dataset),
            rx,
            st,
            sx,
            sy,
            sz,
        };

        let method = self.desc.flush_method;
        self.thread = Some(thread::spawn(move || {
            match method {
                FlushMethod::Sequential => Self::sequential_job(job_desc),
                FlushMethod::Batched(size) => Self::batch_job(job_desc, size),
            }

            tracing::info!("Snapshot flush thread exited");
        }));

        tracing::info!("Snapshot state initialised");

        Ok(())
    }
}

impl Drop for SnapshotState {
    fn drop(&mut self) {
        // Drop sender to signal to thread.
        let _ = self.tx.take();
        tracing::info!("Waiting for snapshots thread to shut down...");
        let now = Instant::now();
        let _ = self.thread.take().map(JoinHandle::join);
        let elapsed = now.elapsed().as_millis();
        tracing::trace!("Time elapsed: {elapsed}");

        if let Err(err) = self.file.as_ref().unwrap().flush() {
            tracing::error!("Failed to flush snapshot file: {err}");
        }

        if let Err(err) = self.file.take().unwrap().close() {
            tracing::error!("Failed to close snapshot file: {err}");
        }
        tracing::info!("Snapshot file closed");
    }
}
