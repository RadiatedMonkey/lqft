use hdf5_metno as hdf5;

/// Creates the HDF5 file and prepares it for storage.
pub fn create_hdf5_file(filename: &str) -> anyhow::Result<hdf5::File> {
    let file = hdf5::File::create(filename)?;

    Ok(file)
}