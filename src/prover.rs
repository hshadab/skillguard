//! ZKML proving wrapper around Jolt Atlas.
//!
//! Provides cryptographic SNARK proofs for skill safety classifications.
//! The MLP forward pass is proved inside the Jolt SNARK VM; softmax
//! stays in plaintext post-proof (only the deterministic MLP is proved).
//!
//! Key types:
//! - [`ProverState`] — initialized once at startup (expensive), shared via `Arc`
//! - [`ProofBundle`] — serializable proof + metadata returned to callers

use std::time::Instant;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use eyre::Result;
use jolt_core::poly::commitment::hyperkzg::HyperKZG;
use onnx_tracer::graph::model::Model;
use onnx_tracer::tensor::Tensor;
use serde::{Deserialize, Serialize};
use tracing::info;
use zkml_jolt_core::snark::JoltSNARK;

use crate::model::skill_safety_model;

/// Maximum trace length for the Jolt prover (2^16 = 65536 steps).
const MAX_TRACE_LENGTH: usize = 1 << 16;

type PCS = HyperKZG<ark_bn254::Bn254>;
type Transcript = jolt_core::utils::transcript::KeccakTranscript;
type F = ark_bn254::Fr;

/// Serializable proof bundle returned from proving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofBundle {
    /// Base64-encoded SNARK proof bytes
    pub proof_b64: String,
    /// Program I/O as JSON (inputs/outputs visible to verifier)
    pub program_io: serde_json::Value,
    /// Size of the serialized proof in bytes
    pub proof_size_bytes: usize,
    /// Time spent proving in milliseconds
    pub proving_time_ms: u64,
}

/// Prover state holding preprocessed keys. Initialize once, share via `Arc`.
pub struct ProverState {
    prover_preprocessing: <JoltSNARK<F, PCS, Transcript> as zkml_jolt_core::snark::SNARK<
        F,
        PCS,
        Transcript,
        fn() -> Model,
    >>::ProverPreprocessing,
    verifier_preprocessing: <JoltSNARK<F, PCS, Transcript> as zkml_jolt_core::snark::SNARK<
        F,
        PCS,
        Transcript,
        fn() -> Model,
    >>::VerifierPreprocessing,
}

// SAFETY: The preprocessing data is read-only after initialization.
// JoltSNARK prove/verify take &self on the preprocessing structs.
unsafe impl Send for ProverState {}
unsafe impl Sync for ProverState {}

impl ProverState {
    /// Initialize prover and verifier preprocessing from the skill safety model.
    ///
    /// This is expensive (~5-15s) and should be done once at startup.
    pub fn initialize() -> Result<Self> {
        let start = Instant::now();
        info!("Initializing ZKML prover (this may take a moment)...");

        let model_fn: fn() -> Model = skill_safety_model;
        let (prover_preprocessing, verifier_preprocessing) =
            <JoltSNARK<F, PCS, Transcript> as zkml_jolt_core::snark::SNARK<
                F,
                PCS,
                Transcript,
                fn() -> Model,
            >>::prover_preprocess(model_fn, MAX_TRACE_LENGTH);

        let elapsed = start.elapsed();
        info!(
            elapsed_ms = elapsed.as_millis() as u64,
            "ZKML prover initialized (Jolt/HyperKZG)"
        );

        Ok(Self {
            prover_preprocessing,
            verifier_preprocessing,
        })
    }

    /// Prove an inference over the given feature vector.
    ///
    /// Returns the proof bundle and the raw model output (4 scores).
    pub fn prove_inference(&self, features: &[i32]) -> Result<(ProofBundle, [i32; 4])> {
        let input = Tensor::new(Some(features), &[1, 22])
            .map_err(|e| eyre::eyre!("Tensor error: {:?}", e))?;

        let start = Instant::now();
        let (snark, program_io) = <JoltSNARK<F, PCS, Transcript> as zkml_jolt_core::snark::SNARK<
            F,
            PCS,
            Transcript,
            fn() -> Model,
        >>::prove(
            &std::slice::from_ref(&input),
            &self.prover_preprocessing,
        );
        let proving_time_ms = start.elapsed().as_millis() as u64;

        // Extract raw output from program IO
        let outputs = &program_io.outputs;
        if outputs.len() < 4 {
            eyre::bail!(
                "Expected at least 4 output values, got {}",
                outputs.len()
            );
        }
        let raw_scores: [i32; 4] = [outputs[0], outputs[1], outputs[2], outputs[3]];

        // Serialize the proof
        let mut proof_bytes = Vec::new();
        snark
            .serialize_compressed(&mut proof_bytes)
            .map_err(|e| eyre::eyre!("Proof serialization failed: {}", e))?;
        let proof_size_bytes = proof_bytes.len();
        let proof_b64 = BASE64.encode(&proof_bytes);

        // Serialize program IO
        let program_io_json = serde_json::to_value(&program_io)
            .map_err(|e| eyre::eyre!("Program IO serialization failed: {}", e))?;

        info!(
            proving_time_ms,
            proof_size_bytes, "ZK proof generated for inference"
        );

        let bundle = ProofBundle {
            proof_b64,
            program_io: program_io_json,
            proof_size_bytes,
            proving_time_ms,
        };

        Ok((bundle, raw_scores))
    }

    /// Verify a proof bundle.
    ///
    /// Returns `Ok(true)` if valid, `Ok(false)` if invalid.
    pub fn verify_proof(&self, bundle: &ProofBundle) -> Result<bool> {
        let proof_bytes = BASE64
            .decode(&bundle.proof_b64)
            .map_err(|e| eyre::eyre!("Base64 decode failed: {}", e))?;

        let snark = <JoltSNARK<F, PCS, Transcript> as CanonicalDeserialize>::deserialize_compressed(
            &proof_bytes[..],
        )
        .map_err(|e| eyre::eyre!("Proof deserialization failed: {}", e))?;

        let program_io: zkml_jolt_core::snark::ProgramIO =
            serde_json::from_value(bundle.program_io.clone())
                .map_err(|e| eyre::eyre!("Program IO deserialization failed: {}", e))?;

        // verify() may panic on invalid proofs, so we catch that
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            <JoltSNARK<F, PCS, Transcript> as zkml_jolt_core::snark::SNARK<
                F,
                PCS,
                Transcript,
                fn() -> Model,
            >>::verify(snark, program_io, &self.verifier_preprocessing)
        }));

        match result {
            Ok(Ok(())) => Ok(true),
            Ok(Err(_)) => Ok(false),
            Err(_) => Ok(false), // panicked = invalid proof
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prover_initialization() {
        let _state = ProverState::initialize().expect("ProverState::initialize() should not fail");
    }

    #[test]
    fn test_prove_and_verify_roundtrip() {
        let state = ProverState::initialize().unwrap();

        // Safe skill features
        let mut features = vec![0i32; 22];
        features[16] = 100; // downloads

        let (bundle, raw_scores) = state.prove_inference(&features).unwrap();

        assert!(bundle.proof_size_bytes > 0);
        assert!(!bundle.proof_b64.is_empty());
        assert!(raw_scores.iter().any(|&s| s != 0));

        // Verify the proof
        let valid = state.verify_proof(&bundle).unwrap();
        assert!(valid, "Valid proof should verify successfully");
    }

    #[test]
    fn test_tampered_proof_fails() {
        let state = ProverState::initialize().unwrap();

        let features = vec![0i32; 22];
        let (mut bundle, _) = state.prove_inference(&features).unwrap();

        // Tamper with the proof by flipping a byte
        let mut proof_bytes = BASE64.decode(&bundle.proof_b64).unwrap();
        if let Some(byte) = proof_bytes.get_mut(10) {
            *byte ^= 0xFF;
        }
        bundle.proof_b64 = BASE64.encode(&proof_bytes);

        let valid = state.verify_proof(&bundle).unwrap();
        assert!(!valid, "Tampered proof should fail verification");
    }
}
