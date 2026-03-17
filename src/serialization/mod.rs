use crate::backend::Backend;
use crate::cost::Cost;
use crate::layer::Layer;
use crate::NeuralNetwork;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::path::Path;

impl<L, C, B> NeuralNetwork<L, C, B>
where
    B: Backend,
    L: Layer<B> + Serialize + for<'de> Deserialize<'de>,
    C: Cost<B>,
{
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let encoded = postcard::to_allocvec(&self.layers)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P, cost: C) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let layers: L = postcard::from_bytes(&buffer)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(NeuralNetwork { layers, cost, _backend: PhantomData })
    }
}
