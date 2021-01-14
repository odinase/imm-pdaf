use super::state_estimator::StateEstimator;
use super::mixture::{ReduceMixture, MixtureParameters};
use super::state_estimator::ekf::GaussParams;
use std::time::{Duration, Instant};
use rayon::prelude::*;

/*
@dataclass
class PDA(Generic[ET]):  # Probabilistic Data Association
    state_filter: StateEstimator[ET]
    clutter_intensity: float
    PD: float
    gate_size: float
*/

pub struct PDAF<S: StateEstimator> {
    state_filter: S,
    clutter_intensity: f64,
    PD: f64,
    gate_size: f64,
}



impl<S> ReduceMixture<<S as StateEstimator>::Params> for PDAF<S>
where
S: StateEstimator + ReduceMixture<<S as StateEstimator>::Params>,
<S as StateEstimator>::Params: std::fmt::Display
{
    /*
    def reduce_mixture(
        self, mixture_filter_state: MixtureParameters[ET]
    ) -> ET:  # the two first moments of the mixture
        """Reduce a Gaussian mixture to a single Gaussian."""
        return self.state_filter.reduce_mixture(mixture_filter_state)
    */
    fn reduce_mixture(&self, mixture_filter_state_weights: &[f64], mixture_filter_state_components: &[<S as StateEstimator>::Params]) -> <S as StateEstimator>::Params {
        self.state_filter.reduce_mixture(mixture_filter_state_weights, mixture_filter_state_components)
    }

}

impl<S> PDAF<S>
where
S: StateEstimator + ReduceMixture<<S as StateEstimator>::Params> + Send + Sync,
<S as StateEstimator>::Params: Clone + std::fmt::Display + Send + Sync,
<S as StateEstimator>::Measurement: Clone + std::fmt::Debug + Send + Sync,
{
    pub fn init(state_filter: S, clutter_intensity: f64, PD: f64, gate_size: f64) -> Self {
        PDAF {
            state_filter,
            clutter_intensity,
            PD,
            gate_size,
        }
    }
    /*
    def predict(self, filter_state: ET, Ts: float) -> ET:
        """Predict state estimate Ts time units ahead"""
        return self.state_filter.predict(filter_state, Ts)
    */
    pub fn predict(&self, filter_state: <S as StateEstimator>::Params, ts: f64) -> <S as StateEstimator>::Params {
        self.state_filter.predict(&filter_state, ts)
    }
    /*
    def gate(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:  # gated, shape=(M,), dtype=bool: gated(j) = true if measurement j is within gate
        """Gate/validate measurements: (z-h(x))'S^(-1)(z-h(x)) <= g^2."""
        gated = np.array(
            [
                self.state_filter.gate(
                    zj,
                    filter_state,
                    sensor_state=sensor_state,
                    gate_size_square=self.gate_size ** 2,
                )
                for zj in Z
            ],
            dtype=bool,
        )
        return gated
    */
        pub fn gate(&self, Z: &[<S as StateEstimator>::Measurement], filter_state: &<S as StateEstimator>::Params) -> Vec<bool> {
        Z.iter().map(
            |z|
            self.state_filter.gate(&z, filter_state, self.gate_size.powi(2))
        ).collect()
    }

    /*
    def loglikelihood_ratios(
        self,  # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:  # shape=(M + 1,), first element for no detection
        """ Calculates the posterior event loglikelihood ratios."""

        log_PD = np.log(self.PD)
        log_PND = np.log(1 - self.PD)  # P_ND = 1 - P_D
        log_clutter = np.log(self.clutter_intensity)

        # allocate
        ll = np.empty(Z.shape[0] + 1)

        # calculate log likelihood ratios
        ll[0] = log_PND + log_clutter  # missed detection
        ll[1:] = np.array(
            [
                self.state_filter.loglikelihood(
                    zj, filter_state, sensor_state=sensor_state
                )
                for zj in Z
            ]
        )
        ll[1:] += log_PD
        return ll
    */
    pub fn loglikelihood_ratios(&self, Z: &[<S as StateEstimator>::Measurement], filter_state: &<S as StateEstimator>::Params) -> Vec<f64> {
        let log_PD = self.PD.ln();
        let log_PND = (1.0 - self.PD).ln();
        let log_clutter = self.clutter_intensity.ln();

        let mut ll = Vec::with_capacity(
            // All associations 
            Z.len() 
            // No association
            + 1
        );

        ll.push(log_PND + log_clutter); // missed detection

        // Loglikelihood of each association
        let log_assos = Z.iter().map(|z| self.state_filter.loglikelihood(&z, filter_state) + log_PD);

        for l in log_assos {
            ll.push(l);
        }
        ll
    }
    /*
    def association_probabilities(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:  # beta, shape=(M + 1,): the association probabilities (normalized likelihood ratios)
        """calculate the poseterior event/association probabilities."""
    
        # log likelihoods
        lls = self.loglikelihood_ratios(Z, filter_state, sensor_state=sensor_state)
    
        # probabilities
        beta = np.exp(lls - scipy.special.logsumexp(lls))
        return beta
    */
    pub fn association_probabilities(&self, Z: &[<S as StateEstimator>::Measurement], filter_state: &<S as StateEstimator>::Params) -> Vec<f64> {
        let lls = self.loglikelihood_ratios(Z, filter_state);
        let logsumexp = lls.iter().map(|l| l.exp()).sum::<f64>().ln();
        let beta = lls.iter().map(|l| (l - logsumexp).exp()).collect();
        beta
    }
    /*
    def conditional_update(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> List[
        ET
    ]:  # Updated filter_state for all association events, first element is no detection
        """Update the state with all possible measurement associations."""
        return [filter_state] + [
            self.state_filter.update(zj, filter_state, sensor_state=sensor_state)
            for zj in Z
        ]
    */
    pub fn conditional_update(&self, Z: &[<S as StateEstimator>::Measurement], filter_state: <S as StateEstimator>::Params) -> Vec<<S as StateEstimator>::Params> {
        let cond_updates = rayon::iter::once(filter_state.clone()).chain(Z.par_iter().map(|z| self.state_filter.update(&z, &filter_state))).collect();

        cond_updates
    }
    /*
    def update(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> ET:  # The filter_state updated by approximating the data association
        """
        Perform the PDA update cycle.
    
        Gate -> association probabilities -> conditional update -> reduce mixture.
        """
        # remove the not gated measurements from consideration
        gated = self.gate(Z, filter_state, sensor_state=sensor_state)
        Zg = Z[gated]
    
        # find association probabilities
        beta = self.association_probabilities(
            Zg, filter_state, sensor_state=sensor_state
        )
    
        # find the mixture components
        filter_state_update_mixture_components = self.conditional_update(
            Zg, filter_state, sensor_state=sensor_state
        )
    
        filter_state_update_mixture = MixtureParameters(
            beta, filter_state_update_mixture_components
        )
        # reduce mixture
        filter_state_updated_reduced = self.reduce_mixture(filter_state_update_mixture)
        return filter_state_updated_reduced
        */
        pub fn update(&self, Z: Vec<<S as StateEstimator>::Measurement>, filter_state: <S as StateEstimator>::Params) -> <S as StateEstimator>::Params {
        let gated = self.gate(Z.as_slice(), &filter_state);
        let Zg: Vec<<S as StateEstimator>::Measurement> = Z.into_par_iter().zip(gated.par_iter()).filter_map(|(z, &g)| if g {Some(z)} else {None}).collect();
            
        // let start_association_probabilities = Instant::now();
        let beta = self.association_probabilities(&Zg, &filter_state);
        // let duration_association_probabilities = start_association_probabilities.elapsed();
        // println!("Time elapsed in association_probabilities is: {:?}", duration_association_probabilities);

        // let start_conditional_update = Instant::now();
        let filter_state_update_mixture_components = self.conditional_update(&Zg, filter_state);
        // let duration_conditional_update = start_conditional_update.elapsed();
        // println!("Time elapsed in conditional_update is: {:?}", duration_conditional_update);
            
        // let start_reduce_mixture = Instant::now();
        let filter_state_update_reduced = self.reduce_mixture(&beta, &filter_state_update_mixture_components);
        // let duration_reduce_mixture = start_reduce_mixture.elapsed();
        // println!("Time elapsed in reduce_mixture is: {:?}", duration_reduce_mixture);

        // let total = (duration_association_probabilities + duration_conditional_update + duration_reduce_mixture).as_nanos() as f32 / 1000.0;

        // println!("Total time: {:.2}\nIn association_probabilities: {:.2}%\nIn conditional_update: {:.2}%\nIn reduce_mixture: {:.2}%", total, duration_association_probabilities.as_nanos() as f32/total*0.1, duration_conditional_update.as_nanos() as f32/total*0.1, duration_reduce_mixture.as_nanos() as f32/total*0.1);
            
        
        filter_state_update_reduced
    }
    /*
        def estimate(self, filter_state: ET) -> GaussParams:
            """Get an estimate with its covariance from the filter state."""
            return self.state_filter.estimate(filter_state)
    */
    pub fn estimate(&self, filter_state: <S as StateEstimator>::Params) -> GaussParams {
        self.state_filter.estimate(filter_state)
    }
}
