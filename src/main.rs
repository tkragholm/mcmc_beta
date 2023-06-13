extern crate plotters;
extern crate rand;
extern crate statrs;

use plotters::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use statrs::distribution::{Beta, Continuous};

/// The beta density function, which describes the probability of a particular state
/// `w` given the parameters `a` and `b`.
///
/// # Arguments
///
/// * `w` - The state for which the beta density is to be calculated.
/// * `a` - The alpha parameter of the beta distribution.
/// * `b` - The beta parameter of the beta distribution.
///
/// # Returns
///
/// The probability density at state `w` according to the beta distribution with
/// parameters `a` and `b`.
fn beta_s(w: f64, a: f64, b: f64) -> f64 {
    w.powf(a - 1.0) * (1.0 - w).powf(b - 1.0)
}

/// A function to simulate the flip of a biased coin, which lands heads with probability `p`.
///
/// # Arguments
///
/// * `p` - The probability of landing heads.
///
/// # Returns
///
/// `true` if the coin lands heads (with probability `p`), `false` otherwise.
fn random_coin(p: f64) -> bool {
    let mut rng = rand::thread_rng();
    let coin: f64 = rng.gen();
    coin < p
}

/// A function to perform Markov Chain Monte Carlo (MCMC) simulations with a beta distribution.
///
/// # Arguments
///
/// * `n_hops` - The number of steps to take in the Markov chain.
/// * `a` - The alpha parameter of the beta distribution.
/// * `b` - The beta parameter of the beta distribution.
///
/// # Returns
///
/// A vector containing the states of the Markov chain after a burn-in period
/// (the first 10% of states are discarded to allow the chain to converge to the target distribution).
fn beta_mcmc(n_hops: usize, a: f64, b: f64) -> Vec<f64> {
    let mut states = vec![];
    let between = Uniform::from(0.0..1.0);
    let mut rng = rand::thread_rng();
    let mut cur = between.sample(&mut rng);
    for _i in 0..n_hops {
        states.push(cur);
        let next = between.sample(&mut rng);
        let ap = f64::min(beta_s(next, a, b) / beta_s(cur, a, b), 1.0);
        if random_coin(ap) {
            cur = next;
        }
    }
    // Return states after burn-in period
    let burn_in = (states.len() as f64 * 0.1) as usize;
    states[states.len() - burn_in..].to_vec() // discard 10% of states
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_hops = 200000; // Number of hops
    let alpha = 2.0; // Alpha parameter
    let beta = 2.0; // Beta parameter
    let states = beta_mcmc(n_hops, alpha, beta); // Generate states

    let bins = 100; // Number of bins for histogram
    let bin_width = 1.0 / bins as f64;
    let mut counts = vec![0; bins]; // Initialize counts

    // Count states in each bin
    for state in &states {
        let bin = (state / bin_width).floor() as usize;
        if bin < bins {
            counts[bin] += 1;
        }
    }

    // Normalize the counts to approximate the PDF
    let total_count = states.len() as f64;
    let counts: Vec<f64> = counts
        .iter()
        .map(|count| *count as f64 / (total_count * bin_width))
        .collect();

    let root = BitMapBackend::new("mcmc.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    // Define the chart
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "MCMC vs Actual Beta Distribution",
            ("Arial", 20).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0f64..1.0f64, 0.0f64..2.5f64)?;

    // Configure the chart
    chart
        .configure_mesh()
        .x_desc("Value")
        .y_desc("Density")
        .draw()?;

    // Draw histogram
    for (i, &count) in counts.iter().enumerate() {
        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (i as f64 * bin_width, 0.0),
                ((i + 1) as f64 * bin_width, count),
            ],
            BLUE.filled(),
        )))?;
    }

    let beta_dist = Beta::new(alpha, beta).unwrap();
    chart.draw_series(LineSeries::new(
        (0..=100).map(|i| {
            let x = i as f64 / 100.0;
            (x, beta_dist.pdf(x))
        }),
        &RED,
    ))?;

    // Manually create elements for the legend
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(0.0, 0.0), (0.1, 0.0)],
            &RED,
        )))
        .unwrap()
        .label(format!("Beta({:.2}, {:.2})", alpha, beta)) // Alpha and Beta inserted here
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(0.0, 0.0), (0.1, 0.0)],
            &BLUE,
        )))
        .unwrap()
        .label(format!("MCMC (n_hops={})", n_hops)) // n_hops inserted here
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Draw the legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
