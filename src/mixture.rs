pub struct MixtureParameters<T> {
    weights: Vec<f64>,
    components: Vec<T>,
}

impl<T> MixtureParameters<T> {
    pub fn new(weights: Vec<f64>, components: Vec<T>) -> Self {
        MixtureParameters {
            weights,
            components,
        }
    }
}

pub trait ReduceMixture<T> {
    fn reduce_mixture(&self, estimator_mixture: MixtureParameters<T>) -> T;
}
