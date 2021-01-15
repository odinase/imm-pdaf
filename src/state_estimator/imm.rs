use super::ekf::GaussParams;
use crate::mixture::{MixtureParameters, ReduceMixture};
use crate::state_estimator::StateEstimator;
use nalgebra::{DMatrix, DMatrixSlice, DVector};

fn discrete_bayes(pr: &[f64], cond_pr: &DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    /*
    Assumes the form
    pr = [s1_k-1, ..., sN_k-1]'
    cond_pr = [[s1_k|s1_k-1, ..., s1_k|sN_k-1]
                    ...
               [sN_k|s1_k-1, ..., sN_k|sN_k-1]]
    */
    let (m, n) = cond_pr.shape();
   /*
      joint = [[s1_k, s1_k-1, ..., s1_k, sN_k-1]
                             ...
               [sN_k, s1_k-1, ..., sN_k, sN_k-1]]
    */
    let joint: DMatrix<f64> = {
        // Broadcast pr rowwise, ie, it repeats at each row for n rows
        let P = DMatrixSlice::from_slice_with_strides(pr, m, n, 1, 0);
        cond_pr.component_mul(&P)
    };
    // marginal = [s1_k, ..., sN_k]'
    let marginal: Vec<f64> = joint.row_sum().iter().cloned().collect();
    /*
    conditional = [[s1_k-1|s1_k, ..., s1_k-1|sN_k]
                              ...
                   [sN_k-1|s1_k, ..., sN_k-1|sN_k]]
    */
    let conditional = {
        // Broadcast pr rowwise, ie, it repeats at each row for n rows
        let M = DMatrixSlice::from_slice_with_strides(marginal.as_slice(), m, n, 0, 1);
        // May blow up, let's hope it doesn't
        joint.component_div(&M)
    };

    let conditional = conditional.transpose();

    (marginal, conditional)
}

struct FlipIter<I>
where
    I: Iterator,
{
    iterators: Vec<I>,
}

impl<I, T> Iterator for FlipIter<I>
where
    I: Iterator<Item = T>,
{
    type Item = Vec<T>;
    fn next(&mut self) -> Option<Self::Item> {
        let output: Option<Vec<T>> = self.iterators.iter_mut().map(|iter| iter.next()).collect();
        output
    }
}

pub struct IMM<S> {
    filters: Vec<S>,
    PI: DMatrix<f64>,
}

impl<S> IMM<S>
where
    S: StateEstimator<Measurement = DVector<f64>>
        + ReduceMixture<<S as StateEstimator>::Params>
        + Clone,
    <S as StateEstimator>::Params: Clone,
{
    pub fn init(filters: Vec<S>, PI: DMatrix<f64>) -> Self {
        IMM { filters, PI }
    }

    /*
    def mix_probabilities(
            self,
            immstate: MixtureParameters[MT],
            # sampling time
            Ts: float,
        ) -> Tuple[
            np.ndarray, np.ndarray
        ]:  # predicted_mode_probabilities, mix_probabilities: shapes = ((M, (M ,M))).
            # mix_probabilities[s] is the mixture weights for mode s
            """Calculate the predicted mode probability and the mixing probabilities."""
            predicted_mode_probabilities, mix_probabilities = discretebayes.discrete_bayes(
                immstate.weights, self.PI
            )
            assert np.all(np.isfinite(predicted_mode_probabilities))
            assert np.all(np.isfinite(mix_probabilities))
            assert np.allclose(mix_probabilities.sum(axis=1), 1)
            return predicted_mode_probabilities, mix_probabilities
    */
    fn mix_probabilities(
        &self,
        immstate: &MixtureParameters<<S as StateEstimator>::Params>,
        _ts: f64,
    ) -> (Vec<f64>, DMatrix<f64>) {
        let (predicted_mode_probabilities, mix_probabilities) =
            discrete_bayes(immstate.weights.as_slice(), &self.PI);
        (predicted_mode_probabilities, mix_probabilities)
    }
    /*
        def mix_states(
            self,
            immstate: MixtureParameters[MT],
            # the mixing probabilities: shape=(M, M)
            mix_probabilities: np.ndarray,
        ) -> List[MT]:
            mixed_states = [
                fs.reduce_mixture(MixtureParameters(mix_pr_s, immstate.components))
                for fs, mix_pr_s in zip(self.filters, mix_probabilities)
            ]
            return mixed_states
    */
    fn mix_states(
        &self,
        immstate_components: &[<S as StateEstimator>::Params],
        mix_probabilities: DMatrix<f64>,
    ) -> Vec<<S as StateEstimator>::Params> {
        let mixed_states = self
            .filters
            .iter()
            .zip(
                mix_probabilities
                    .row_iter() // Iterate over rows
                    .map(|r| r.into_owned())
            )
            .map(|(fs, mix_pr_s)| {
                fs.reduce_mixture(
                    mix_pr_s.as_slice(),
                    immstate_components
                )
            })
            .collect();
        mixed_states
    }
    /*
        def mode_matched_prediction(
            self,
            mode_states: List[MT],
            # The sampling time
            Ts: float,
        ) -> List[MT]:
            modestates_pred = [
                fs.predict(cs, Ts) for fs, cs in zip(self.filters, mode_states)
            ]
            return modestates_pred
    */
    fn mode_matched_prediction(
        &self,
        mode_states: Vec<<S as StateEstimator>::Params>,
        ts: f64,
    ) -> Vec<<S as StateEstimator>::Params> {
        let modestates_pred = self
            .filters
            .iter()
            .zip(mode_states.into_iter())
            .map(|(fs, cs)| fs.predict(&cs, ts))
            .collect();
        modestates_pred
    }
    /*
    def mode_matched_update(
        self,
            z: np.ndarray,
            immstate: MixtureParameters[MT],
            sensor_state: Optional[Dict[str, Any]] = None,
        ) -> List[MT]:
        """Update each mode in immstate with z in sensor_state."""
            updated_state = [
                fs.update(z, cs, sensor_state=sensor_state)
                for fs, cs in zip(self.filters, immstate.components)
                ]
                return updated_state
    */
    fn mode_matched_update(
        &self,
        z: &DVector<f64>,
        immstate: &MixtureParameters<<S as StateEstimator>::Params>,
    ) -> Vec<<S as StateEstimator>::Params> {
        let updated_state = self
            .filters
            .iter()
            .zip(immstate.components.iter())
            .map(|(fs, cs)| fs.update(&z, cs))
            .collect();
        updated_state
    }
    /*
    def update_mode_probabilities(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Dict[str, Any] = None,
    ) -> np.ndarray:
    """Calculate the mode probabilities in immstate updated with z in sensor_state"""
    loglikelihood = np.array(
            [
                fs.loglikelihood(z, cs, sensor_state=sensor_state)
                for fs, cs in zip(self.filters, immstate.components)
                ]
            )
            logjoint = loglikelihood + np.log(immstate.weights)
            updated_mode_probabilities = np.exp(logjoint - logsumexp(logjoint))
            assert np.all(np.isfinite(updated_mode_probabilities))
            assert np.allclose(np.sum(updated_mode_probabilities), 1)
        return updated_mode_probabilities
        */
    fn update_mode_probabilities(
        &self,
        z: &DVector<f64>,
        immstate: &MixtureParameters<<S as StateEstimator>::Params>,
    ) -> Vec<f64> {
        let logjoint: Vec<f64> = self
            .filters
            .iter()
            .zip(immstate.components.iter())
            .zip(immstate.weights.iter())
            .map(|((fs, cs), w)| fs.loglikelihood(&z, cs) + w.ln())
            .collect();

        let logsumexp = logjoint.iter().map(|l| l.exp()).sum::<f64>().ln();

        let updated_mode_probabilities = logjoint
            .into_iter()
            .map(|l| (l - logsumexp).exp())
            .collect();
        updated_mode_probabilities
    }
}

impl<S> StateEstimator for IMM<S>
where
    S: StateEstimator<Measurement = DVector<f64>>
        + ReduceMixture<<S as StateEstimator>::Params>
        + Clone,
    <S as StateEstimator>::Params: Clone,
{
    type Params = MixtureParameters<<S as StateEstimator>::Params>;
    type Measurement = DVector<f64>;

    /*
        def predict(
        self,
        immstate: MixtureParameters[MT],
        # sampling time
        Ts: float,
    ) -> MixtureParameters[MT]:
    """
    Predict the immstate Ts time units ahead approximating the mixture step.
    Ie. Predict mode probabilities, condition states on predicted mode,
    appoximate resulting state distribution as Gaussian for each mode, then predict each mode.
        """
        predicted_mode_probability, mixing_probability = self.mix_probabilities(
            immstate, Ts
        )
        mixed_mode_states: List[MT] = self.mix_states(immstate, mixing_probability)
        predicted_mode_states = self.mode_matched_prediction(mixed_mode_states, Ts)
        predicted_immstate = MixtureParameters(
            predicted_mode_probability, predicted_mode_states
        )
        return predicted_immstate
        */
    fn predict(&self, immstate: &Self::Params, ts: f64) -> Self::Params {
        let (predicted_mode_probability, mixing_probability) =
            self.mix_probabilities(&immstate, ts);

        let mixed_mode_states = self.mix_states(immstate.components.as_slice(), mixing_probability);
        let predicted_mode_states = self.mode_matched_prediction(mixed_mode_states, ts);

        let predicted_immstate =
            MixtureParameters::new(predicted_mode_probability, predicted_mode_states);

        predicted_immstate
    }

    /*
        def update(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Dict[str, Any] = None,
    ) -> MixtureParameters[MT]:
    """Update the immstate with z in sensor_state."""
    updated_weights = self.update_mode_probabilities(
        z, immstate, sensor_state=sensor_state
    )
    updated_states = self.mode_matched_update(
            z, immstate, sensor_state=sensor_state
        )
        updated_immstate = MixtureParameters(updated_weights, updated_states)
        return updated_immstate
        */
    fn update(&self, z: &Self::Measurement, immstate: &Self::Params) -> Self::Params {
        let updated_weights = self.update_mode_probabilities(z, &immstate);

        let updated_states = self.mode_matched_update(z, immstate);

        let updated_immstate = MixtureParameters::new(updated_weights, updated_states);

        updated_immstate
    }

    /*
    def step(
            self,
            z,
            immstate: MixtureParameters[MT],
            Ts: float,
            sensor_state: Dict[str, Any] = None,
        ) -> MixtureParameters[MT]:
        """Predict immstate with Ts time units followed by updating it with z in sensor_state"""
        predicted_immstate = self.predict(immstate, Ts)
        updated_immstate = self.update(z, predicted_immstate, sensor_state=sensor_state)
        return updated_immstate
        */
    fn step(&self, z: &Self::Measurement, immstate: &Self::Params, ts: f64) -> Self::Params {
        let predicted_immstate = self.predict(immstate, ts);
        let updated_immstate = self.update(z, &predicted_immstate);

        updated_immstate
    }

    /*
                def estimate(self, immstate: MixtureParameters[MT]) -> GaussParams:
                """Calculate a state estimate with its covariance from immstate"""
                # ! assuming all the modes have the same reduce and estimate
                dataRed = self.filters[0].reduce_mixture(immstate)
                return self.filters[0].estimate(dataRed)
    */
    fn estimate(&self, immstate: Self::Params) -> GaussParams {
        let (weights, components) = immstate.destructure();
        let reduced_estimate = self.filters[0].reduce_mixture(weights.as_slice(), components.as_slice());
        self.filters[0].estimate(reduced_estimate)
    }
    /*
        def loglikelihood(
            self,
            z: np.ndarray,
            immstate: MixtureParameters,
            *,
            sensor_state: Dict[str, Any] = None,
        ) -> float:
        mode_conditioned_ll = np.fromiter(
        (
            fs.loglikelihood(z, modestate_s, sensor_state=sensor_state)
            for fs, modestate_s in zip(self.filters, immstate.components)
        ),
        dtype=float,
    )
    ll = logsumexp(mode_conditioned_ll, b=immstate.weights)
    return ll
    */
    fn loglikelihood(&self, z: &Self::Measurement, immstate: &Self::Params) -> f64 {
        let ll = self
            .filters
            .iter()
            .zip(immstate.iter())
            .map(|(fs, (weight, modestate_s))| (fs.loglikelihood(z, &modestate_s).exp() * weight))
            .sum::<f64>()
            .ln();

        ll
    }

    /*
        def gate(
            self,
            z: np.ndarray,
            immstate: MixtureParameters[MT],
            gate_size_square: float,
            sensor_state: Dict[str, Any] = None,
    ) -> bool:
    """Check if z is within the gate of any mode in immstate in sensor_state"""
    gated_per = [
        fs.gate(z, ds, gate_size_square, sensor_state=sensor_state)
        for fs, ds in zip(self.filters, immstate.components)
        ]
        gated = any(gated_per)
        return gated
        */
    fn gate(&self, z: &Self::Measurement, immstate: &Self::Params, gate_size_square: f64) -> bool {
        let gated = self
            .filters
            .iter()
            .zip(immstate.components.iter())
            .any(|(fs, ds)| fs.gate(z, ds, gate_size_square));
        gated
    }
}

impl<S> ReduceMixture<MixtureParameters<<S as StateEstimator>::Params>> for IMM<S>
where
    S: StateEstimator + ReduceMixture<<S as StateEstimator>::Params>,
    MixtureParameters<<S as StateEstimator>::Params>: Clone,
{
    /*
    def reduce_mixture(
        self, immstate_mixture: MixtureParameters[MixtureParameters[MT]]
    ) -> MixtureParameters[MT]:
    """Approximate a mixture of immstates as a single immstate"""
    # extract probabilities as array
    weights = immstate_mixture.weights
    component_conditioned_mode_prob = np.array(
        [c.weights.ravel() for c in immstate_mixture.components]
    )
    # flip conditioning order with Bayes
    mode_prob, mode_conditioned_component_prob = discretebayes.discrete_bayes(
        weights, component_conditioned_mode_prob
    )
    # from list of component containing lists of modes
    # to list of modes containting lists of components
    comps_per_mode = zip(*[comp.components for comp in immstate_mixture.components])
    mode_states = [
        fs.reduce_mixture(MixtureParameters(mode_s_cond_comp_prob, mode_s_comp))
        for fs, mode_s_cond_comp_prob, mode_s_comp in zip(
            self.filters, mode_conditioned_component_prob, comps_per_mode,
        )
        ]
        immstate_reduced = MixtureParameters(mode_prob, mode_states)
        return immstate_reduced
        */
    fn reduce_mixture(
        &self,
        immstate_weights: &[f64],
        immstate_components: &[MixtureParameters<<S as StateEstimator>::Params>]
    ) -> MixtureParameters<<S as StateEstimator>::Params> {
        let m = immstate_components[0].weights.len();
        let n = immstate_components.len();

        let component_conditioned_mode_prob = DMatrix::from_iterator(
            m,
            n,
            immstate_components
                .iter()
                // Don't really see a way of avoiding the copy here. Needs to ponder more
                .cloned()
                .map(|c| c.weights)
                .flatten(),
        )
        .transpose();

        let (mode_prob, mode_conditioned_component_prob) =
            discrete_bayes(immstate_weights, &component_conditioned_mode_prob);

        /*
        Some pseudo code:
        comps_per_mode = Vec with length of num modes, containing a Vec for all possible estimates for that one state given an association (from previous iteration)
        for i, s in enumerate(modes) {
            for k, a in enumerate(assos) {
                comps_per_mode[i].push(immstate.components[k].components[i])
            }
        }
        */
        let comps_per_mode = FlipIter {
            iterators: immstate_components
                .iter()
                .cloned()
                .map(|v| v.components.into_iter())
                .collect(),
        };

        let mode_states = self
            .filters
            .iter()
            .zip(
                mode_conditioned_component_prob
                    .row_iter()
                    .map(|a| a.into_owned())
            ) // Convert RowVector to Vec
            .zip(comps_per_mode)
            .map(|((fs, mode_s_cond_comp_prob), mode_s_comp)| {
                fs.reduce_mixture(mode_s_cond_comp_prob.as_slice(), &mode_s_comp)
            })
            .collect();

        let immstate_reduced = MixtureParameters::new(mode_prob, mode_states);

        immstate_reduced
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_estimator::ekf::EKF;
    use crate::state_estimator::models::{DynamicModel, MeasurementModel};

    use std::f64::consts::PI;
    static SIGMA_A_CV: f64 = 0.2;
    static SIGMA_A_CT: f64 = 0.1;
    static SIGMA_OMEGA: f64 = 0.002 * PI;
    static SIGMA_Z: f64 = 3.0;
    static TS: f64 = 2.5;

    #[test]
    fn test_imm_predict() {
        let x = DVector::from_row_slice(&[0., 0., 0., 0., 0.]);
        let P = DMatrix::from_diagonal(&DVector::from_row_slice(&[1., 1., 1., 1., 1.]));

        let measmod = MeasurementModel::CartesianPosition(SIGMA_Z);
        let dynmod_ct = DynamicModel::CT(SIGMA_A_CT, SIGMA_OMEGA);
        let dynmod_cv = DynamicModel::CV(SIGMA_A_CV);

        let ekf_cv = EKF::init(dynmod_cv, measmod);

        // Is probably moved, so make a new one
        let measmod = MeasurementModel::CartesianPosition(SIGMA_Z);

        let ekf_ct = EKF::init(dynmod_ct, measmod);

        let filters = vec![ekf_cv, ekf_ct];

        let cv_state = GaussParams::new(x, P);
        let ct_state = cv_state.clone();
        let components = vec![cv_state, ct_state];
        let weights = vec![0.5, 0.5];
        let immstate = MixtureParameters::new(weights, components);

        let trans_prob_mat = DMatrix::from_row_slice(2, 2, &[0.95, 0.05, 0.05, 0.95]);

        let imm = IMM::init(filters, trans_prob_mat);
        let immstate = imm.predict(&immstate, TS);
        let estimate = imm.estimate(immstate);

        let x_correct = DVector::from_row_slice(&[0., 0., 0., 0., 0.]);
        let P_correct = DMatrix::from_row_slice(
            5,
            5,
            &[
                7.38020833, 0., 2.578125, 0., 0., 0., 7.38020833, 0., 2.578125, 0., 2.578125, 0.,
                1.0625, 0., 0., 0., 2.578125, 0., 1.0625, 0., 0., 0., 0., 0., 0.50004935,
            ],
        );
        assert!(estimate.x.len() == 5, "x.len() = {}", estimate.x.len());
        assert!(
            estimate.P.shape() == (5, 5),
            "P.shape() = {:#?}",
            estimate.P.shape()
        );
        assert!(x_correct.relative_eq(&estimate.x, 1e-5, 1e-5));
        assert!(P_correct.relative_eq(&estimate.P, 1e-5, 1e-5));
    }

    #[test]
    fn test_imm_step() {
        let x = DVector::from_row_slice(&[0., 0., 0., 0., 0.]);
        let P = DMatrix::from_diagonal(&DVector::from_row_slice(&[1., 1., 1., 1., 1.]));
        let z = DVector::from_row_slice(&[2.46850281, 24.68253298]);

        let measmod = MeasurementModel::CartesianPosition(SIGMA_Z);
        let dynmod_ct = DynamicModel::CT(SIGMA_A_CT, SIGMA_OMEGA);
        let dynmod_cv = DynamicModel::CV(SIGMA_A_CV);

        let ekf_cv = EKF::init(dynmod_cv, measmod);

        // Is probably moved, so make a new one
        let measmod = MeasurementModel::CartesianPosition(SIGMA_Z);

        let ekf_ct = EKF::init(dynmod_ct, measmod);

        let filters = vec![ekf_cv, ekf_ct];

        let cv_state = GaussParams::new(x, P);
        let ct_state = cv_state.clone();
        let components = vec![cv_state, ct_state];
        let weights = vec![0.5, 0.5];
        let immstate = MixtureParameters::new(weights, components);

        let trans_prob_mat = DMatrix::from_row_slice(2, 2, &[0.95, 0.05, 0.05, 0.95]);

        let imm = IMM::init(filters, trans_prob_mat);
        let immstate = imm.predict(&immstate, TS);
        let immstate = imm.update(&z, &immstate);
        let estimate = imm.estimate(immstate);
        let x_correct =
            DVector::from_row_slice(&[1.11271634, 11.12603866, 0.38894036, 3.88901048, 0.]);
        let P_correct = DMatrix::from_row_slice(
            5,
            5,
            &[
                4.05693263, 0.00041544, 1.41808462, 0.00033466, 0., 0.00041544, 4.06104508,
                0.00033466, 1.42139743, 0., 1.41808462, 0.00033466, 0.65876418, 0.00026959, 0.,
                0.00033466, 1.42139743, 0.00026959, 0.66143283, 0., 0., 0., 0., 0., 0.45773908,
            ],
        );
        assert!(estimate.x.len() == 5, "x.len() = {}", estimate.x.len());
        assert!(
            estimate.P.shape() == (5, 5),
            "P.shape() = {:#?}",
            estimate.P.shape()
        );
        assert!(x_correct.relative_eq(&estimate.x, 1e-5, 1e-5));
        assert!(P_correct.relative_eq(&estimate.P, 1e-5, 1e-5));
    }
}
