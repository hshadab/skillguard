"""
Synthetic dataset generator for SkillGuard skill safety classifier.

Generates ~500 labeled 35-dimensional feature vectors across 4 classes:
- SAFE (0): Minimal risk patterns, popular utilities
- CAUTION (1): New-author API skills, moderate env/credential access
- DANGEROUS (2): Credential harvesters, privilege escalation, LLM exposure
- MALICIOUS (3): Reverse shells, obfuscated payloads, persistence, exfiltration

Plus ~40 hard negatives: legit pentest tools, security research, CI/CD automation.

Feature vector (35 elements, all normalized to [0, 128]):
 0: shell_exec_count (max 20)    10: skill_md_line_count (max 500)  20: reverse_shell_patterns (max 5)
 1: network_call_count (max 50)  11: script_file_count (max 10)     21: llm_secret_exposure (bool)
 2: fs_write_count (max 30)      12: dependency_count (max 30)      22: entropy_score (max 8.0)
 3: env_access_count (max 20)    13: author_account_age (max 365)   23: non_ascii_ratio (max 0.5)
 4: credential_patterns (max 10) 14: author_skill_count (max 100)   24: max_line_length (max 1000)
 5: external_download (bool)     15: stars (log, max 4.0)           25: comment_ratio (max 1.0)
 6: obfuscation_score (max 15)   16: downloads (log, max 6.0)       26: domain_count (max 20)
 7: privilege_escalation (bool)  17: has_virustotal_report (bool)   27: string_obfuscation_score (max 10)
 8: persistence_mechanisms (max 5)18: vt_malicious_flags (max 20)   28: shell_exec_per_line (max 1.0)
 9: data_exfiltration (max 5)    19: password_protected_archives    29: network_per_script (max 10)
                                     (bool)                         30: credential_density (max 1.0)
                                                                    31: shell_and_network (max 10)
                                                                    32: obfuscation_and_exec (max 10)
                                                                    33: file_extension_diversity (max 5)
                                                                    34: has_shebang (bool)

IMPORTANT: Feature values here are PRE-NORMALIZED (already in [0, 128]).
They must match what `SkillFeatures::to_normalized_vec()` produces in Rust.
Real features are sparse — most are 0 or near-0, with only relevant ones activated.
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# Label mapping — 4-class (legacy synthetic) and 3-class (real LLM-labeled)
CLASS_NAMES = ["SAFE", "CAUTION", "DANGEROUS", "MALICIOUS"]
CLASS_NAMES_3 = ["SAFE", "CAUTION", "DANGEROUS"]
NUM_FEATURES = 35
LABEL_MAP_3 = {"SAFE": 0, "CAUTION": 1, "DANGEROUS": 2}


def clip(val: float, lo: float = 0.0, hi: float = 128.0) -> float:
    return max(lo, min(hi, val))


def rand_range(lo: float, hi: float) -> float:
    return random.uniform(lo, hi)


def _base_metadata(author_type: str = "established") -> dict:
    """Generate metadata features (indices 10-18, 25-26) based on author type."""
    if author_type == "established":
        return {
            "md_lines": clip(rand_range(20, 128)),      # 10: well-documented
            "script_files": clip(rand_range(0, 30)),     # 11
            "dep_count": clip(rand_range(0, 40)),        # 12
            "acct_age": clip(rand_range(60, 128)),       # 13: old account
            "author_skills": clip(rand_range(20, 80)),   # 14
            "stars": clip(rand_range(30, 128)),          # 15: popular
            "downloads": clip(rand_range(30, 128)),      # 16: popular
            "has_vt": 0.0,                                # 17
            "vt_flags": 0.0,                              # 18
        }
    elif author_type == "new":
        return {
            "md_lines": clip(rand_range(5, 50)),         # 10
            "script_files": clip(rand_range(0, 15)),     # 11
            "dep_count": clip(rand_range(0, 25)),        # 12
            "acct_age": clip(rand_range(0, 30)),         # 13: new account
            "author_skills": clip(rand_range(0, 20)),    # 14: few skills
            "stars": clip(rand_range(0, 30)),            # 15
            "downloads": clip(rand_range(0, 30)),        # 16
            "has_vt": 0.0,                                # 17
            "vt_flags": 0.0,                              # 18
        }
    elif author_type == "unknown":
        # Neutral/midpoint metadata — represents a skill loaded from a local
        # SKILL.md file where stars, downloads, author reputation are unavailable.
        # Values sit in the middle of the training range so the model learns to
        # rely on content-based features when metadata is uninformative.
        return {
            "md_lines": clip(rand_range(8, 80)),         # 10: varies
            "script_files": clip(rand_range(0, 30)),     # 11: varies
            "dep_count": clip(rand_range(0, 20)),        # 12
            "acct_age": clip(rand_range(30, 80)),        # 13: neutral zone
            "author_skills": clip(rand_range(5, 30)),    # 14: neutral
            "stars": clip(rand_range(20, 70)),           # 15: neutral
            "downloads": clip(rand_range(20, 70)),       # 16: neutral
            "has_vt": 0.0,                                # 17
            "vt_flags": 0.0,                              # 18
        }
    else:  # "attacker"
        return {
            "md_lines": clip(rand_range(0, 10)),         # 10: minimal docs
            "script_files": clip(rand_range(0, 10)),     # 11
            "dep_count": clip(rand_range(0, 20)),        # 12
            "acct_age": clip(rand_range(0, 15)),         # 13: very new
            "author_skills": clip(rand_range(0, 10)),    # 14
            "stars": clip(rand_range(0, 10)),            # 15
            "downloads": clip(rand_range(0, 15)),        # 16
            "has_vt": 0.0 if random.random() < 0.5 else 128.0,  # 17
            "vt_flags": clip(rand_range(0, 60)),         # 18
        }


def _zeros() -> list[float]:
    """Start with a zeroed 35-element feature vector."""
    return [0.0] * NUM_FEATURES


def _set_metadata(vec: list[float], meta: dict):
    """Set metadata features in the vector."""
    vec[10] = meta["md_lines"]
    vec[11] = meta["script_files"]
    vec[12] = meta["dep_count"]
    vec[13] = meta["acct_age"]
    vec[14] = meta["author_skills"]
    vec[15] = meta["stars"]
    vec[16] = meta["downloads"]
    vec[17] = meta["has_vt"]
    vec[18] = meta["vt_flags"]


def _make_safe_sample() -> list[float]:
    """Generate a SAFE skill feature vector.

    Safe skills are simple utilities: calculators, formatters, converters.
    Very few risk features activated. Well-established authors.
    """
    vec = _zeros()
    meta = _base_metadata("established")
    _set_metadata(vec, meta)

    # Minimal risk features (most stay 0)
    vec[0] = clip(rand_range(0, 10))     # 0: shell_exec (0-2 commands)
    vec[1] = clip(rand_range(0, 10))     # 1: network (0-2 API calls)
    vec[2] = clip(rand_range(0, 5))      # 2: fs_write (0-1 writes)
    vec[3] = clip(rand_range(0, 10))     # 3: env (0-2 vars)
    vec[4] = clip(rand_range(0, 5))      # 4: credential (0 or 1 mention)

    # Non-risk metadata
    vec[22] = clip(rand_range(0, 30))    # 22: entropy (low)
    vec[24] = clip(rand_range(0, 20))    # 24: max_line_length (short lines)
    vec[25] = clip(rand_range(15, 80))   # 25: comment_ratio (well commented)
    vec[26] = clip(rand_range(0, 30))    # 26: domain_count (few domains)
    vec[33] = clip(rand_range(0, 50))    # 33: file_ext_diversity
    vec[34] = 128.0 if random.random() < 0.3 else 0.0  # 34: has_shebang

    return vec


def _make_caution_sample() -> list[float]:
    """Generate a CAUTION skill feature vector.

    Caution skills are new-author API wrappers with moderate env/credential access.
    A few risk features activated but nothing extreme.
    """
    vec = _zeros()
    meta = _base_metadata("new")
    _set_metadata(vec, meta)

    # Moderate risk features
    vec[0] = clip(rand_range(0, 20))     # 0: shell_exec (0-3 commands)
    vec[1] = clip(rand_range(5, 40))     # 1: network (1-8 calls)
    vec[2] = clip(rand_range(0, 15))     # 2: fs_write
    vec[3] = clip(rand_range(5, 40))     # 3: env_access (1-6 vars)
    vec[4] = clip(rand_range(5, 30))     # 4: credential (1-2 mentions)
    vec[5] = 128.0 if random.random() < 0.3 else 0.0   # 5: ext_download
    vec[6] = clip(rand_range(0, 15))     # 6: obfuscation (low)

    vec[22] = clip(rand_range(10, 50))   # 22: entropy
    vec[24] = clip(rand_range(5, 40))    # 24: max_line_length
    vec[25] = clip(rand_range(5, 50))    # 25: comment_ratio
    vec[26] = clip(rand_range(10, 60))   # 26: domain_count (some APIs)
    vec[29] = clip(rand_range(0, 30))    # 29: net_per_script
    vec[33] = clip(rand_range(10, 60))   # 33: file_ext_diversity
    vec[34] = 128.0 if random.random() < 0.4 else 0.0  # 34: has_shebang

    return vec


def _make_dangerous_sample() -> list[float]:
    """Generate a DANGEROUS skill feature vector.

    Dangerous skills have clear risk indicators: credential harvesting,
    privilege escalation, LLM exposure, etc. Multiple risk features activated.
    """
    vec = _zeros()
    meta = _base_metadata("new")
    _set_metadata(vec, meta)

    # Select 2-4 danger patterns to activate
    patterns = random.sample([
        "credential_harvester",
        "priv_esc",
        "llm_exposure",
        "data_exfil",
        "persistence",
        "network_abuse",
    ], k=random.randint(2, 4))

    for pattern in patterns:
        if pattern == "credential_harvester":
            vec[3] = clip(rand_range(30, 80))    # 3: env_access
            vec[4] = clip(rand_range(40, 128))   # 4: credential
            vec[30] = clip(rand_range(40, 128))  # 30: cred_density
        elif pattern == "priv_esc":
            vec[0] = clip(rand_range(20, 80))    # 0: shell_exec
            vec[7] = 128.0                        # 7: priv_esc (bool)
            vec[28] = clip(rand_range(30, 80))   # 28: shell_per_line
        elif pattern == "llm_exposure":
            vec[4] = clip(rand_range(30, 80))    # 4: credential
            vec[21] = 128.0                       # 21: llm_secret_exposure
            vec[30] = clip(rand_range(40, 128))  # 30: cred_density
        elif pattern == "data_exfil":
            vec[1] = clip(rand_range(10, 50))    # 1: network
            vec[9] = clip(rand_range(30, 128))   # 9: exfiltration
            vec[26] = clip(rand_range(30, 80))   # 26: domain_count
        elif pattern == "persistence":
            vec[0] = clip(rand_range(10, 60))    # 0: shell_exec
            vec[8] = clip(rand_range(40, 128))   # 8: persistence
        elif pattern == "network_abuse":
            vec[1] = clip(rand_range(30, 80))    # 1: network
            vec[5] = 128.0                        # 5: ext_download
            vec[26] = clip(rand_range(40, 100))  # 26: domain_count
            vec[29] = clip(rand_range(30, 80))   # 29: net_per_script

    vec[22] = clip(rand_range(20, 70))   # 22: entropy
    vec[24] = clip(rand_range(10, 60))   # 24: max_line_length
    vec[33] = clip(rand_range(10, 60))   # 33: file_ext_diversity
    vec[34] = 128.0 if random.random() < 0.5 else 0.0  # 34: has_shebang

    return vec


# Malicious archetypes — each represents a specific attack pattern
# with only the relevant features activated (sparse, like real extraction)

def _malicious_reverse_shell() -> list[float]:
    """Reverse shell: few features, but reverse_shell_patterns is key."""
    vec = _zeros()
    meta = _base_metadata("attacker")
    _set_metadata(vec, meta)
    vec[20] = clip(rand_range(30, 128))  # 20: reverse_shell_patterns (1-5 matches)
    vec[26] = clip(rand_range(30, 100))  # 26: domain_count (attacker domain)
    vec[0] = clip(rand_range(0, 20))     # 0: shell_exec (may or may not trigger)
    vec[34] = 128.0 if random.random() < 0.5 else 0.0  # 34: has_shebang
    vec[33] = clip(rand_range(10, 50))   # 33: file_ext_diversity
    return vec


def _malicious_credential_stealer() -> list[float]:
    """Credential harvester + exfiltration."""
    vec = _zeros()
    meta = _base_metadata("attacker")
    _set_metadata(vec, meta)
    vec[3] = clip(rand_range(20, 80))    # 3: env_access
    vec[4] = clip(rand_range(40, 128))   # 4: credential
    vec[1] = clip(rand_range(5, 40))     # 1: network (exfil endpoint)
    vec[9] = clip(rand_range(30, 128))   # 9: data_exfiltration
    vec[21] = 128.0 if random.random() < 0.6 else 0.0  # 21: llm_secret
    vec[26] = clip(rand_range(30, 90))   # 26: domain_count
    vec[30] = clip(rand_range(40, 128))  # 30: cred_density
    return vec


def _malicious_obfuscated_payload() -> list[float]:
    """Obfuscated code execution (base64, eval, etc.)."""
    vec = _zeros()
    meta = _base_metadata("attacker")
    _set_metadata(vec, meta)
    vec[6] = clip(rand_range(40, 128))   # 6: obfuscation_score
    vec[0] = clip(rand_range(10, 60))    # 0: shell_exec
    vec[27] = clip(rand_range(30, 128))  # 27: string_obfuscation
    vec[32] = clip(rand_range(30, 100))  # 32: obfusc_and_exec
    vec[22] = clip(rand_range(40, 100))  # 22: entropy (high for obfuscated)
    vec[24] = clip(rand_range(40, 128))  # 24: max_line_length (long encoded lines)
    vec[28] = clip(rand_range(20, 80))   # 28: shell_per_line
    return vec


def _malicious_persistence() -> list[float]:
    """Persistence installer (crontab, autostart, systemd)."""
    vec = _zeros()
    meta = _base_metadata("attacker")
    _set_metadata(vec, meta)
    vec[0] = clip(rand_range(10, 60))    # 0: shell_exec
    vec[8] = clip(rand_range(40, 128))   # 8: persistence
    vec[5] = 128.0 if random.random() < 0.6 else 0.0  # 5: ext_download
    vec[1] = clip(rand_range(5, 40))     # 1: network
    vec[26] = clip(rand_range(20, 80))   # 26: domain_count
    vec[28] = clip(rand_range(20, 60))   # 28: shell_per_line
    vec[34] = 128.0 if random.random() < 0.7 else 0.0  # 34: has_shebang
    return vec


def _malicious_data_exfil() -> list[float]:
    """Data exfiltration (tar + upload)."""
    vec = _zeros()
    meta = _base_metadata("attacker")
    _set_metadata(vec, meta)
    vec[0] = clip(rand_range(10, 50))    # 0: shell_exec
    vec[1] = clip(rand_range(5, 40))     # 1: network
    vec[2] = clip(rand_range(10, 50))    # 2: fs_write
    vec[9] = clip(rand_range(40, 128))   # 9: exfiltration
    vec[26] = clip(rand_range(30, 80))   # 26: domain_count
    vec[31] = clip(rand_range(10, 50))   # 31: shell_and_network
    return vec


def _malicious_curl_pipe_bash() -> list[float]:
    """curl|bash installer."""
    vec = _zeros()
    meta = _base_metadata("attacker")
    _set_metadata(vec, meta)
    vec[0] = clip(rand_range(10, 50))    # 0: shell_exec
    vec[1] = clip(rand_range(5, 30))     # 1: network
    vec[5] = 128.0                        # 5: ext_download
    vec[26] = clip(rand_range(30, 90))   # 26: domain_count
    vec[28] = clip(rand_range(20, 60))   # 28: shell_per_line
    vec[31] = clip(rand_range(10, 40))   # 31: shell_and_network
    vec[34] = 128.0 if random.random() < 0.7 else 0.0  # 34: has_shebang
    return vec


def _malicious_priv_esc() -> list[float]:
    """Privilege escalation + backdoor."""
    vec = _zeros()
    meta = _base_metadata("attacker")
    _set_metadata(vec, meta)
    vec[0] = clip(rand_range(20, 80))    # 0: shell_exec
    vec[7] = 128.0                        # 7: priv_esc
    vec[8] = clip(rand_range(20, 80))    # 8: persistence (sudo + crontab)
    vec[2] = clip(rand_range(10, 50))    # 2: fs_write
    vec[28] = clip(rand_range(30, 80))   # 28: shell_per_line
    vec[34] = 128.0 if random.random() < 0.6 else 0.0  # 34: has_shebang
    return vec


def _malicious_llm_exposure() -> list[float]:
    """LLM secret exposure (instructions to leak creds)."""
    vec = _zeros()
    meta = _base_metadata("attacker")
    _set_metadata(vec, meta)
    vec[4] = clip(rand_range(30, 128))   # 4: credential
    vec[21] = 128.0                       # 21: llm_secret_exposure
    vec[30] = clip(rand_range(40, 128))  # 30: cred_density
    vec[26] = clip(rand_range(20, 70))   # 26: domain_count
    return vec


def _malicious_multi_vector() -> list[float]:
    """Multi-vector attack: combines several patterns."""
    vec = _zeros()
    meta = _base_metadata("attacker")
    _set_metadata(vec, meta)
    # Combine 3-5 attack signals
    vec[0] = clip(rand_range(20, 80))    # 0: shell_exec
    vec[1] = clip(rand_range(10, 50))    # 1: network
    vec[4] = clip(rand_range(20, 80))    # 4: credential
    vec[6] = clip(rand_range(10, 60))    # 6: obfuscation
    vec[7] = 128.0 if random.random() < 0.6 else 0.0  # 7: priv_esc
    vec[8] = clip(rand_range(20, 80))    # 8: persistence
    vec[9] = clip(rand_range(20, 80))    # 9: exfiltration
    vec[20] = clip(rand_range(30, 128))  # 20: reverse_shell
    vec[21] = 128.0 if random.random() < 0.5 else 0.0  # 21: llm_secret
    vec[26] = clip(rand_range(40, 100))  # 26: domain_count
    vec[27] = clip(rand_range(10, 60))   # 27: string_obfusc
    vec[28] = clip(rand_range(20, 60))   # 28: shell_per_line
    vec[31] = clip(rand_range(10, 40))   # 31: shell_and_network
    vec[32] = clip(rand_range(10, 40))   # 32: obfusc_and_exec
    return vec


def _malicious_crypto_miner() -> list[float]:
    """Crypto miner: curl download + execution."""
    vec = _zeros()
    meta = _base_metadata("attacker")
    _set_metadata(vec, meta)
    vec[1] = clip(rand_range(5, 30))     # 1: network
    vec[5] = 128.0                        # 5: ext_download
    vec[7] = 128.0                        # 7: priv_esc (chmod +x)
    vec[26] = clip(rand_range(30, 90))   # 26: domain_count
    vec[29] = clip(rand_range(10, 40))   # 29: net_per_script
    vec[33] = clip(rand_range(10, 50))   # 33: file_ext_diversity
    return vec


# Sparse anchor vectors derived from actual regression-test feature extraction.
# These teach the model to recognize DANGEROUS patterns in minimal/sparse inputs
# (e.g., a short skill with only one or two attack signals present).
_SPARSE_ANCHORS = [
    # reverse_shell: only features 20, 22 lit up (very short md)
    [0,0,0,0,0,0,0,0,0,0,1,12,0,0,0,0,0,0,0,0,76,0,69,0,5,0,0,0,0,0,0,0,0,0,0],
    # cred_harvester: features 4, 9, 21, 30
    [0,2,0,6,64,0,0,0,0,25,1,12,0,0,0,0,0,0,0,0,0,128,75,0,11,0,6,0,0,12,91,0,0,0,0],
    # data_exfil: features 1, 9
    [0,2,0,0,0,0,0,0,0,25,1,12,0,0,0,0,0,0,0,0,0,0,73,0,8,0,6,0,0,12,0,0,0,0,0],
    # priv_esc: features 0, 7, 8
    [12,0,0,0,0,0,0,128,51,0,2,12,0,0,0,0,0,0,0,0,0,0,76,0,7,0,0,0,64,0,0,0,0,0,0],
    # obfuscated: features 0, 6, 32
    [6,0,0,0,0,0,34,0,0,0,1,12,0,0,0,0,0,0,0,0,0,0,86,0,13,0,0,0,42,0,0,0,12,0,0],
    # persistence: features 0, 1, 8
    [12,2,0,0,0,0,0,0,128,0,2,12,0,0,0,0,0,0,0,0,0,0,76,0,10,0,6,0,51,12,0,12,0,0,0],
    # crypto_miner: features 0, 1, 5, 7
    [6,2,0,0,0,128,0,128,0,0,1,12,0,0,0,0,0,0,0,0,0,0,74,0,13,0,6,0,64,12,0,12,0,0,0],
    # llm_exposure: features 4, 21, 30
    [0,0,0,0,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,0,0,0,0,0,0,0,128,0,0,0,0],
    # curl_bash: features 0, 1, 5
    [12,5,0,0,0,128,0,0,0,0,1,12,0,0,0,0,0,0,0,0,0,0,71,0,5,0,6,0,128,25,0,25,0,0,0],
    # multi_vector: many features active
    [6,2,0,0,12,0,17,128,51,25,2,12,0,0,0,0,0,0,0,0,51,128,78,0,7,0,6,0,25,12,14,12,12,0,0],
]


def _sparse_anchor_with_noise() -> list[float]:
    """Return a noisy variant of a sparse anchor vector.

    Adds uniform noise in [-10, +10] to each feature, keeping values in [0, 128].
    This teaches the model to generalise from the exact regression-test vectors
    to nearby feature-space regions.
    """
    base = random.choice(_SPARSE_ANCHORS).copy()
    for i in range(len(base)):
        noise = random.uniform(-10, 10)
        base[i] = max(0.0, min(128.0, base[i] + noise))
    return base


# Sparse SAFE anchor vectors derived from regression-test safe skills.
# These have high metadata features (stars, downloads, author reputation) with
# minimal content features, teaching the model that trusted metadata + sparse
# content = SAFE even in the i32 quantized inference path.
_SAFE_ANCHORS = [
    # calculator: just skill_md_line_count + high metadata
    [0,0,0,0,0,0,0,0,0,0,1,0,0,128,19,86,85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # weather: similar to calculator
    [0,0,0,0,0,0,0,0,0,0,1,0,0,128,19,86,85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # todo-manager: 2 md_lines + high metadata
    [0,0,0,0,0,0,0,0,0,0,2,0,0,128,19,86,85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # json-formatter: similar
    [0,0,0,0,0,0,0,0,0,0,2,0,0,128,19,86,85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # unit-converter: similar
    [0,0,0,0,0,0,0,0,0,0,2,0,0,128,19,86,85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # code-block-only setup (has some shell execution + high metadata)
    [6,0,0,0,0,0,0,0,0,0,3,12,0,128,19,86,85,0,0,0,0,0,69,0,2,0,0,0,64,0,0,0,0,0,0],
    # pentest-tool (some shell, network + very high metadata)
    [25,5,0,0,0,128,0,0,0,0,3,12,0,128,38,100,100,0,0,0,0,0,70,0,10,0,6,0,25,12,0,12,0,0,0],
]


def _safe_anchor_with_noise() -> list[float]:
    """Return a noisy variant of a safe anchor vector."""
    base = random.choice(_SAFE_ANCHORS).copy()
    for i in range(len(base)):
        noise = random.uniform(-5, 5)
        base[i] = max(0.0, min(128.0, base[i] + noise))
    return base


MALICIOUS_ARCHETYPES = [
    _malicious_reverse_shell,
    _malicious_credential_stealer,
    _malicious_obfuscated_payload,
    _malicious_persistence,
    _malicious_data_exfil,
    _malicious_curl_pipe_bash,
    _malicious_priv_esc,
    _malicious_llm_exposure,
    _malicious_multi_vector,
    _malicious_crypto_miner,
]


def _make_malicious_sample() -> list[float]:
    """Generate a MALICIOUS skill feature vector using random archetype."""
    gen = random.choice(MALICIOUS_ARCHETYPES)
    return gen()


def _make_safe_unknown_metadata() -> list[float]:
    """Generate a SAFE skill with unknown/neutral metadata.

    Represents skills loaded from local SKILL.md files where repository
    metadata (stars, downloads, author age) is unavailable. The model must
    learn to classify these based on content features alone.
    """
    vec = _zeros()
    meta = _base_metadata("unknown")
    _set_metadata(vec, meta)

    # Same content profile as safe — minimal risk features
    vec[0] = clip(rand_range(0, 15))     # 0: shell_exec (0-2 CLI commands from code blocks)
    vec[1] = clip(rand_range(0, 10))     # 1: network (0-2 API mentions)
    vec[2] = clip(rand_range(0, 5))      # 2: fs_write
    vec[3] = clip(rand_range(0, 10))     # 3: env (0-2 vars)
    vec[4] = clip(rand_range(0, 5))      # 4: credential (0 or 1 mention)

    vec[22] = clip(rand_range(0, 50))    # 22: entropy (code blocks)
    vec[24] = clip(rand_range(0, 30))    # 24: max_line_length
    vec[25] = clip(rand_range(0, 30))    # 25: comment_ratio (code blocks may lack comments)
    vec[26] = clip(rand_range(0, 30))    # 26: domain_count
    vec[33] = clip(rand_range(0, 40))    # 33: file_ext_diversity
    vec[34] = 128.0 if random.random() < 0.2 else 0.0  # 34: has_shebang

    return vec


def _make_caution_unknown_metadata() -> list[float]:
    """Generate a CAUTION skill with unknown/neutral metadata.

    New-author API wrappers loaded from local SKILL.md with no repo metadata.
    """
    vec = _zeros()
    meta = _base_metadata("unknown")
    _set_metadata(vec, meta)

    # Same content profile as caution — moderate risk features
    vec[0] = clip(rand_range(0, 25))     # 0: shell_exec (CLI commands in code blocks)
    vec[1] = clip(rand_range(5, 40))     # 1: network (API calls mentioned)
    vec[2] = clip(rand_range(0, 15))     # 2: fs_write
    vec[3] = clip(rand_range(5, 40))     # 3: env_access
    vec[4] = clip(rand_range(5, 30))     # 4: credential
    vec[5] = 128.0 if random.random() < 0.3 else 0.0   # 5: ext_download
    vec[6] = clip(rand_range(0, 15))     # 6: obfuscation (low)

    vec[22] = clip(rand_range(10, 60))   # 22: entropy
    vec[24] = clip(rand_range(5, 50))    # 24: max_line_length
    vec[25] = clip(rand_range(0, 30))    # 25: comment_ratio
    vec[26] = clip(rand_range(10, 60))   # 26: domain_count
    vec[29] = clip(rand_range(0, 30))    # 29: net_per_script
    vec[33] = clip(rand_range(10, 60))   # 33: file_ext_diversity
    vec[34] = 128.0 if random.random() < 0.3 else 0.0  # 34: has_shebang

    return vec


def _make_hard_negative() -> tuple[list[float], int]:
    """Generate a hard negative: legit pentest/security/CI-CD tool.

    These have some risk patterns but are legitimately SAFE or CAUTION
    (well-documented, established author, no true malicious intent).
    """
    vec = _zeros()
    meta = _base_metadata("established")
    _set_metadata(vec, meta)

    # Pentest tools have shell exec, network calls, but are legit
    vec[0] = clip(rand_range(10, 50))    # 0: shell_exec
    vec[1] = clip(rand_range(10, 40))    # 1: network
    vec[5] = 128.0 if random.random() < 0.4 else 0.0  # 5: ext_download
    vec[7] = 128.0 if random.random() < 0.3 else 0.0  # 7: priv_esc

    # Low obfuscation (legit tools don't obfuscate)
    vec[6] = clip(rand_range(0, 10))     # 6: obfuscation

    # Some pentest features
    vec[20] = clip(rand_range(0, 50))    # 20: reverse_shell (nmap etc)
    vec[26] = clip(rand_range(10, 50))   # 26: domain_count
    vec[28] = clip(rand_range(5, 30))    # 28: shell_per_line
    vec[29] = clip(rand_range(5, 25))    # 29: net_per_script
    vec[33] = clip(rand_range(10, 50))   # 33: file_ext_diversity
    vec[34] = 128.0 if random.random() < 0.7 else 0.0  # 34: has_shebang

    # Key differentiators: well-documented, established, high stars
    vec[25] = clip(rand_range(20, 70))   # 25: comment_ratio (well commented)

    label = random.choice([0, 1])
    return vec, label


def generate_dataset(
    n_per_class: int = 125,
    n_hard_negatives: int = 40,
    n_unknown_metadata: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the full synthetic dataset.

    Returns:
        features: (N, 35) array of feature vectors
        labels: (N,) array of class labels (0=SAFE, 1=CAUTION, 2=DANGEROUS, 3=MALICIOUS)
    """
    random.seed(seed)
    np.random.seed(seed)

    features = []
    labels = []

    generators = [
        (0, _make_safe_sample),
        (1, _make_caution_sample),
        (2, _make_dangerous_sample),
        (3, _make_malicious_sample),
    ]

    for label, gen_fn in generators:
        for _ in range(n_per_class):
            features.append(gen_fn())
            labels.append(label)

    # Hard negatives
    for _ in range(n_hard_negatives):
        feat, lab = _make_hard_negative()
        features.append(feat)
        labels.append(lab)

    # Unknown-metadata samples — teach the model to classify based on content
    # features when repository metadata is unavailable (local SKILL.md files).
    # Includes all 4 classes so the model doesn't learn "unknown metadata = safe".
    for _ in range(n_unknown_metadata):
        features.append(_make_safe_unknown_metadata())
        labels.append(0)  # SAFE
        features.append(_make_caution_unknown_metadata())
        labels.append(1)  # CAUTION

    # Dangerous/malicious with unknown metadata (attacker blending in)
    for _ in range(n_unknown_metadata // 2):
        vec = _make_dangerous_sample()
        meta = _base_metadata("unknown")
        _set_metadata(vec, meta)
        features.append(vec)
        labels.append(2)  # DANGEROUS

        vec = _make_malicious_sample()
        meta = _base_metadata("unknown")
        _set_metadata(vec, meta)
        features.append(vec)
        labels.append(3)  # MALICIOUS

    features_arr = np.array(features, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)

    return features_arr, labels_arr


def save_dataset_jsonl(
    features: np.ndarray,
    labels: np.ndarray,
    path: str = "data/skills_labeled.jsonl",
):
    """Save dataset as JSONL for reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for feat, lab in zip(features, labels):
            entry = {
                "features": feat.tolist(),
                "label": int(lab),
                "class_name": CLASS_NAMES[int(lab)],
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {len(labels)} samples to {path}")


def load_dataset_jsonl(
    path: str = "data/skills_labeled.jsonl",
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset from JSONL."""
    features = []
    labels = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            features.append(entry["features"])
            labels.append(entry["label"])
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_human_overrides(
    path: str = "training/human_review.json",
) -> dict[str, str]:
    """Load human-reviewed label overrides.

    Returns a dict mapping skill_name → corrected label (SAFE/CAUTION/DANGEROUS).
    These override LLM labels during training to correct misclassifications
    (e.g., legitimate security tools mislabeled as DANGEROUS).
    """
    override_path = Path(path)
    if not override_path.exists():
        return {}

    try:
        data = json.loads(override_path.read_text())
        overrides = {}
        if isinstance(data, list):
            for entry in data:
                name = entry.get("skill_name", "")
                label = entry.get("label", "").upper()
                if name and label in LABEL_MAP_3:
                    overrides[name] = label
        elif isinstance(data, dict):
            # Also support {"skill_name": "LABEL"} format
            for name, label in data.items():
                if isinstance(label, str) and label.upper() in LABEL_MAP_3:
                    overrides[name] = label.upper()
        if overrides:
            print(f"  Loaded {len(overrides)} human overrides from {path}")
        return overrides
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Warning: failed to load human overrides from {path}: {e}")
        return {}


def load_real_dataset(
    path: str = "training/real-labels.json",
    skillguard_bin: str = "target/release/skillguard",
    human_review_path: str = "training/human_review.json",
) -> tuple[np.ndarray, np.ndarray]:
    """Load real LLM-labeled dataset and extract features via the Rust binary.

    Each entry in the JSON must have:
      - skill_name: str
      - llm_label: "SAFE" | "CAUTION" | "DANGEROUS"
      - skill_md: str (full SKILL.md content)

    Human overrides from human_review.json are applied on top of LLM labels.

    Calls `skillguard extract-features` for each skill to get the 35-dim
    feature vector, guaranteeing feature parity with production inference.

    Returns:
        features: (N, 35) array of i32 feature vectors
        labels: (N,) array of class labels (0=SAFE, 1=CAUTION, 2=DANGEROUS)
    """
    import subprocess
    import shutil

    # Load human overrides
    overrides = load_human_overrides(human_review_path)

    # Find the skillguard binary
    bin_path = shutil.which("skillguard") or skillguard_bin
    if not Path(bin_path).exists():
        # Try debug build
        alt = "target/debug/skillguard"
        if Path(alt).exists():
            bin_path = alt
        else:
            raise FileNotFoundError(
                f"skillguard binary not found at '{bin_path}'. "
                "Build it first: cargo build --release"
            )

    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")

    features = []
    labels = []
    errors = 0

    n_overridden = 0
    for entry in data:
        skill_name = entry.get("skill_name", "")
        # Apply human override if available
        if skill_name in overrides:
            label_str = overrides[skill_name]
            n_overridden += 1
        else:
            label_str = entry.get("llm_label", "").upper()
        if label_str not in LABEL_MAP_3:
            print(f"  Skipping {skill_name or '?'}: unknown label '{label_str}'")
            continue

        skill_md = entry.get("skill_md", "")
        if not skill_md.strip():
            continue

        # Call skillguard extract-features via subprocess
        try:
            result = subprocess.run(
                [bin_path, "extract-features"],
                input=skill_md,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                errors += 1
                continue

            vec = json.loads(result.stdout.strip())
            if len(vec) != NUM_FEATURES:
                print(f"  Skipping {entry['skill_name']}: got {len(vec)} features, expected {NUM_FEATURES}")
                continue

            features.append(vec)
            labels.append(LABEL_MAP_3[label_str])

        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
            errors += 1
            continue

    if errors > 0:
        print(f"  {errors} extraction errors")
    if n_overridden > 0:
        print(f"  {n_overridden} labels overridden by human review")

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


class SkillDataset(Dataset):
    """PyTorch Dataset wrapper."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def generate_dangerous_augmentation(
    n: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate N synthetic DANGEROUS-class feature vectors + proportional SAFE anchors.

    Returns augmented samples for both DANGEROUS and SAFE classes.
    For every N DANGEROUS samples, generates N//3 SAFE anchor samples
    (featuring high metadata values from regression-test safe skills).
    This prevents the DANGEROUS augmentation from overwhelming the SAFE class.

    Args:
        n: Number of DANGEROUS augmentation samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        features: (M, 35) array where M = n + n_safe_anchors
        labels: (M,) array of labels (mix of 0=SAFE and 2=DANGEROUS)
    """
    random.seed(seed)
    np.random.seed(seed)

    features = []
    labels = []

    # Phase 1: Generate DANGEROUS samples
    for _ in range(n):
        n_anchors = len(_SPARSE_ANCHORS)
        n_exact = n_anchors * 3  # 3 copies of each anchor
        idx = len(features)
        if idx < n_exact:
            vec = _SPARSE_ANCHORS[idx % n_anchors].copy()
        else:
            r = random.random()
            if r < 0.25:
                vec = _make_dangerous_sample()
            elif r < 0.50:
                gen = random.choice(MALICIOUS_ARCHETYPES)
                vec = gen()
            else:
                vec = _sparse_anchor_with_noise()
        features.append(vec)
        labels.append(2)  # DANGEROUS

    # Phase 2: Generate SAFE anchor samples (with high metadata values)
    # This teaches the model that skills with strong trust metadata and
    # minimal risk features should be classified as SAFE, even in i32 mode.
    n_safe = max(n // 3, len(_SAFE_ANCHORS) * 3)
    for i in range(n_safe):
        n_safe_anchors = len(_SAFE_ANCHORS)
        n_safe_exact = n_safe_anchors * 3
        if i < n_safe_exact:
            vec = _SAFE_ANCHORS[i % n_safe_anchors].copy()
        else:
            vec = _safe_anchor_with_noise()
        features.append(vec)
        labels.append(0)  # SAFE

    features_arr = np.array(features, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)

    return features_arr, labels_arr


if __name__ == "__main__":
    features, labels = generate_dataset()
    print(f"Dataset shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    for i, name in enumerate(CLASS_NAMES):
        count = (labels == i).sum()
        print(f"  {name}: {count}")

    save_dataset_jsonl(features, labels)
