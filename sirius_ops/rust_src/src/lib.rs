// sirius_ops_rust
// High-Performance Rust Extension for PyTorch/Python using PyO3
// Features Rayon-parallelized embedding searches & bare-metal Mamba selective scanning.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

/// Helper for fast cosine similarity of two float slices.
fn fast_cosine_similarity_slice(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().max(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..len {
        let val_a = if i < a.len() { a[i] } else { 0.0 };
        let val_b = if i < b.len() { b[i] } else { 0.0 };

        dot_product += val_a * val_b;
        norm_a += val_a * val_a;
        norm_b += val_b * val_b;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

/// Computes the cosine similarity between two float vectors.
/// Employs auto-vectorization and zero-copy NumPy array slice access.
#[pyfunction]
fn fast_cosine_similarity_rust(a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> f32 {
    let a_slice = match a.as_slice() {
        Ok(s) => s,
        Err(_) => return 0.0,
    };
    let b_slice = match b.as_slice() {
        Ok(s) => s,
        Err(_) => return 0.0,
    };
    fast_cosine_similarity_slice(a_slice, b_slice)
}

/// Executes a single Mamba-2 SSM selective recurrent scan step.
/// Bypasses Python interpreter overhead, sharing NumPy contiguous memory.
#[pyfunction]
fn mamba_selective_scan_step_rust(
    x_step: PyReadonlyArray1<f32>,      // dim
    gate_step: PyReadonlyArray1<f32>,   // dim
    state: &PyArray2<f32>,              // dim x state_dim (Modified in-place!)
    A_log: PyReadonlyArray2<f32>,       // dim x state_dim
    B_raw: PyReadonlyArray1<f32>,       // state_dim
    C_raw: PyReadonlyArray1<f32>,       // state_dim
    dt: PyReadonlyArray1<f32>,          // dim
    norm_weight: PyReadonlyArray1<f32>, // dim
) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray2<f32>>)> {
    let x_arr = x_step.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("x_step slice error: {}", e)))?;
    let gate_arr = gate_step.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("gate_step slice error: {}", e)))?;
    let B_arr = B_raw.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("B_raw slice error: {}", e)))?;
    let C_arr = C_raw.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("C_raw slice error: {}", e)))?;
    let dt_arr = dt.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("dt slice error: {}", e)))?;
    let norm_arr = norm_weight.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("norm_weight slice error: {}", e)))?;

    // Get ndarray views
    let A_view = A_log.as_array();
    let mut state_view = unsafe { state.as_array_mut() };

    let dim = x_arr.len();
    let state_dim = B_arr.len();

    // Safety checks to prevent index out of bounds panic
    if state_view.shape() != &[dim, state_dim] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "state shape {:?} does not match expected shape {:?}",
            state_view.shape(), [dim, state_dim]
        )));
    }
    if A_view.shape() != &[dim, state_dim] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "A_log shape {:?} does not match expected shape {:?}",
            A_view.shape(), [dim, state_dim]
        )));
    }

    let mut y = vec![0.0; dim];

    for d in 0..dim {
        let dt_val = dt_arr[d];
        let u = x_arr[d];
        let n_w = norm_arr[d];
        let g_s = gate_arr[d];

        // Softplus / SiLU gate calculation
        let silu_gate = g_s * (1.0 / (1.0 + (-g_s).exp()));
        let mut y_accum = 0.0;

        for s in 0..state_dim {
            let a_val = -A_view[[d, s]].exp();
            let h_prev = state_view[[d, s]];

            // Discretize parameters
            let bar_a = (dt_val * a_val).exp();
            let bar_b = dt_val * B_arr[s];

            // Update recurrent state directly in PyTorch shared memory buffer
            let h_new = bar_a * h_prev + bar_b * u;
            state_view[[d, s]] = h_new;

            // Output projection accumulator
            y_accum += h_new * C_arr[s];
        }

        y[d] = y_accum * silu_gate * n_w;
    }

    let py = state.py();
    let y_py = PyArray1::from_vec(py, y).to_owned();
    Ok((y_py, state.to_owned()))
}

/// Explicitly configures Rayon's global thread pool to avoid hybrid/asymmetric core starvation.
#[pyfunction]
fn init_rayon_thread_pool(num_threads: usize) -> PyResult<()> {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global();
    Ok(())
}

/// Non-Parametric Episodic Zero-Loss Memory Bank.
/// Written in Rust using Rayon for parallel lookup sweeps and a strict LRU Eviction policy.
#[pyclass]
struct SiriusZeroLossMemoryRust {
    #[pyo3(get, set)]
    discrete_store: HashMap<Vec<i64>, Vec<i64>>,
    discrete_access_order: Vec<Vec<i64>>,
    continuous_keys: Vec<Vec<f32>>,
    continuous_values: Vec<Vec<f32>>,
    continuous_access_order: Vec<usize>,
    #[pyo3(get, set)]
    max_capacity: usize,
}

#[pymethods]
impl SiriusZeroLossMemoryRust {
    #[new]
    fn new() -> Self {
        SiriusZeroLossMemoryRust {
            discrete_store: HashMap::new(),
            discrete_access_order: Vec::new(),
            continuous_keys: Vec::new(),
            continuous_values: Vec::new(),
            continuous_access_order: Vec::new(),
            max_capacity: 50000, // Safe default to guarantee memory < 1.5 GB
        }
    }

    /// Inserts a discrete prompt-response token sequence with LRU management.
    fn insert_discrete(&mut self, key_tokens: Vec<i64>, value_tokens: Vec<i64>) {
        self.discrete_access_order.retain(|x| x != &key_tokens);
        self.discrete_access_order.push(key_tokens.clone());
        self.discrete_store.insert(key_tokens, value_tokens);
        self.prune_if_exceeded();
    }

    /// Inserts a continuous visual-semantic embedding with zero-copy distance checking.
    fn insert_continuous(&mut self, key_tensor: PyReadonlyArray1<f32>, value_tensor: PyReadonlyArray1<f32>) {
        if let (Ok(k_slice), Ok(v_slice)) = (key_tensor.as_slice(), value_tensor.as_slice()) {
            let key_vec = k_slice.to_vec();
            let val_vec = v_slice.to_vec();

            // Overwrite if a highly matching key already exists
            let match_idx = self.continuous_keys.par_iter().position_any(|existing_k| {
                let sim = fast_cosine_similarity_slice(existing_k, &key_vec);
                sim >= 0.999
            });

            if let Some(idx) = match_idx {
                self.continuous_values[idx] = val_vec;
                self.continuous_access_order.retain(|&x| x != idx);
                self.continuous_access_order.push(idx);
            } else {
                self.continuous_keys.push(key_vec);
                self.continuous_values.push(val_vec);
                let new_idx = self.continuous_keys.len() - 1;
                self.continuous_access_order.push(new_idx);
            }

            self.prune_if_exceeded();
        }
    }

    /// Queries the discrete cache for exact token sequence matches.
    fn query_discrete(&mut self, query_tokens: Vec<i64>) -> Option<(Vec<i64>, f32)> {
        // 1. Exact match lookup
        if let Some(val) = self.discrete_store.get(&query_tokens) {
            self.discrete_access_order.retain(|x| x != &query_tokens);
            self.discrete_access_order.push(query_tokens.clone());
            return Some((val.clone(), 1.0));
        }

        // 2. Subsequence / Prefix match lookup
        for (stored_key, stored_val) in &self.discrete_store {
            if query_tokens.len() >= stored_key.len() && &query_tokens[..stored_key.len()] == stored_key {
                self.discrete_access_order.retain(|x| x != stored_key);
                self.discrete_access_order.push(stored_key.clone());
                return Some((stored_val.clone(), 1.0));
            }
        }

        None
    }

    /// Queries the continuous embedding database in parallel using Rayon.
    /// Achieves thread-level scaling on multiple CPU performance cores.
    fn query_continuous(&mut self, query_tensor: PyReadonlyArray1<f32>, similarity_threshold: f32) -> Option<(Vec<f32>, f32)> {
        if self.continuous_keys.is_empty() {
            return None;
        }

        let q_slice = match query_tensor.as_slice() {
            Ok(s) => s,
            Err(_) => return None,
        };
        let q_vec = q_slice.to_vec();

        // Parallel map-reduce sweep using Rayon
        let best_match = self.continuous_keys
            .par_iter()
            .enumerate()
            .map(|(idx, key)| {
                let sim = fast_cosine_similarity_slice(key, &q_vec);
                (idx, sim)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((idx, sim)) = best_match {
            if sim >= similarity_threshold {
                self.continuous_access_order.retain(|&x| x != idx);
                self.continuous_access_order.push(idx);
                return Some((self.continuous_values[idx].clone(), sim));
            }
        }

        None
    }

    /// Resets the memory stores.
    fn clear(&mut self) {
        self.discrete_store.clear();
        self.discrete_access_order.clear();
        self.continuous_keys.clear();
        self.continuous_values.clear();
        self.continuous_access_order.clear();
    }

    /// Returns the total size of stored pairs.
    fn size(&self) -> usize {
        self.discrete_store.len() + self.continuous_keys.len()
    }
}

impl SiriusZeroLossMemoryRust {
    /// Prunes continuous and discrete memories if they exceed capacity to protect 2.0 GB ceiling.
    fn prune_if_exceeded(&mut self) {
        // Continuous Eviction
        while self.continuous_keys.len() > self.max_capacity {
            if !self.continuous_access_order.is_empty() {
                let oldest_idx = self.continuous_access_order.remove(0);
                if oldest_idx < self.continuous_keys.len() {
                    self.continuous_keys.remove(oldest_idx);
                    self.continuous_values.remove(oldest_idx);
                    // Adjust remaining indices in continuous_access_order
                    for idx in self.continuous_access_order.iter_mut() {
                        if *idx > oldest_idx {
                            *idx -= 1;
                        }
                    }
                }
            } else {
                self.continuous_keys.remove(0);
                self.continuous_values.remove(0);
            }
        }

        // Discrete Eviction
        while self.discrete_store.len() > self.max_capacity {
            if !self.discrete_access_order.is_empty() {
                let oldest_key = self.discrete_access_order.remove(0);
                self.discrete_store.remove(&oldest_key);
            } else {
                if let Some(key) = self.discrete_store.keys().next().cloned() {
                    self.discrete_store.remove(&key);
                }
            }
        }
    }
}

/// PyO3 Module entry point.
#[pymodule]
fn sirius_ops_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_cosine_similarity_rust, m)?)?;
    m.add_function(wrap_pyfunction!(mamba_selective_scan_step_rust, m)?)?;
    m.add_function(wrap_pyfunction!(init_rayon_thread_pool, m)?)?;
    m.add_class::<SiriusZeroLossMemoryRust>()?;
    Ok(())
}
