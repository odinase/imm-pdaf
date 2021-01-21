use crate::state_estimator::ekf::GaussParams;
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct MixtureParameters<T> {
    pub weights: Vec<f64>,
    pub components: Vec<T>,
}

impl<T> std::fmt::Display for MixtureParameters<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = String::new();
        for (w, c) in self.iter() {
            d.push_str(&format!("\nweight: {}\ncomponent: {}", w, c));
        }
        write!(f, "{}", d)
    }
}

impl<'a, T> MixtureParameters<T> {
    pub fn new(weights: Vec<f64>, components: Vec<T>) -> Self {
        MixtureParameters {
            weights,
            components,
        }
    }

    pub fn iter(&'a self) -> MixParamsIter<'a, T> {
        self.into_iter()
    }

    pub fn iter_mut(&'a mut self) -> MixParamsIterMut<'a, T> {
        self.into_iter()
    }

    pub fn destructure(self) -> (Vec<f64>, Vec<T>) {
        (self.weights, self.components)
    }
}

pub struct MixParamsIntoIter<T> {
    elem: std::iter::Zip<std::vec::IntoIter<f64>, std::vec::IntoIter<T>>,
}

pub struct MixParamsIter<'a, T> {
    elem: std::iter::Zip<std::slice::Iter<'a, f64>, std::slice::Iter<'a, T>>,
}

pub struct MixParamsIterMut<'a, T> {
    elem: std::iter::Zip<std::slice::IterMut<'a, f64>, std::slice::IterMut<'a, T>>,
}

impl<T> IntoIterator for MixtureParameters<T> {
    type Item = (f64, T);
    type IntoIter = MixParamsIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let elem = self.weights.into_iter().zip(self.components.into_iter());
        MixParamsIntoIter { elem }
    }
}

impl<'a, T> IntoIterator for &'a MixtureParameters<T> {
    type Item = (&'a f64, &'a T);
    type IntoIter = MixParamsIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        let elem = self.weights.iter().zip(self.components.iter());
        MixParamsIter { elem }
    }
}

impl<'a, T> IntoIterator for &'a mut MixtureParameters<T> {
    type Item = (&'a mut f64, &'a mut T);
    type IntoIter = MixParamsIterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        let elem = self.weights.iter_mut().zip(self.components.iter_mut());
        MixParamsIterMut { elem }
    }
}

impl<T> Iterator for MixParamsIntoIter<T> {
    type Item = (f64, T);
    fn next(&mut self) -> Option<Self::Item> {
        self.elem.next()
    }
}

impl<'a, T> Iterator for MixParamsIter<'a, T> {
    type Item = (&'a f64, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        self.elem.next()
    }
}

impl<'a, T> Iterator for MixParamsIterMut<'a, T> {
    type Item = (&'a mut f64, &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        self.elem.next()
    }
}

// Maybe seems odd to not use MixtureParameters here as input, but this is due to Rust ownership
pub trait ReduceMixture<T> {
    fn reduce_mixture(&self, weights: &[f64], components: &[T]) -> T;
}

pub fn gaussian_reduce_mixture(
    weights: &[f64],
    components: &[GaussParams],
) -> (DVector<f64>, DMatrix<f64>) {
    // We assume all components have equal state length
    let nx = components[0].x.len();
    let x_mean = components.iter().map(|p| &p.x).zip(weights.iter()).fold(
        DVector::zeros(nx),
        |mut xmean, (x, &w)| {
            xmean += x * w;
            xmean
        },
    );
    let P_mean = components
        .iter()
        .map(|p| (&p.x, &p.P))
        .zip(weights.iter())
        .fold(DMatrix::zeros(nx, nx), |mut P_mean, ((x, P), &w)| {
            let xdiff = x - &x_mean;
            P_mean += P * w + w * &xdiff * &xdiff.transpose();
            P_mean
        });
    (x_mean, P_mean)
}
