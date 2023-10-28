use serde::Deserialize;
use std::{
    fs::File,
    path::Path
};

#[derive(Debug, Clone, Deserialize)]
pub struct Timestep {
    pub Ts: f64,
    pub Z: Vec<f64>,
    pub Z_shape: (usize, usize),
    pub Xgt: Vec<f64>,
}

pub fn read_dataset_from_json(data_path: impl AsRef<Path>) -> anyhow::Result<Vec<Timestep>> {
    let file = File::open(data_path)?;
    let timesteps = serde_json::from_reader(file)?;

    Ok(timesteps)
}
