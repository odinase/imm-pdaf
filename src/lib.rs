pub mod state_estimator;
pub mod mixture;

pub mod yoyo {
    pub fn testy() {
        println!("testst");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
