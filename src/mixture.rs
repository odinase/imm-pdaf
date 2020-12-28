pub mod Mixture {
    pub struct MixtureParameters<T> {
        weights: Vec<f64>,
        components: Vec<T>,
    }

    pub trait ReduceMixture<T> {
        fn reduce_mixture(self, estimator_mixture: MixtureParameters<T>) -> T;
    }
}