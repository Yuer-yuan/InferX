# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Policy extracted from weifang231/eb-vllm; exact source pin in manifest.json.
from __future__ import annotations
import atexit
import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
logger = logging.getLogger(__name__)
_PD_THETA_HARD_MIN = 0.01

@dataclass
class HardwareParams:
    """Hardware timing parameters for the P/D Competition Scheduler.

    The timing models are:
        Prefill: T_p = α_p + β_p × L  (L = input tokens)
        Decode:  T_d = α_d + β_d × k  (k = batch size)

    All time values are in seconds.
    """
    alpha_p: float
    beta_p: float
    alpha_d: float
    beta_d: float
    model: str = ''
    device_name: str = ''
    dtype: str = 'float16'
    timestamp: str = ''
    measurement_type: str = 'decomposed'
    prefill_r2: float = 0.0
    decode_r2: float = 0.0
    prefill_sizes: list[int] = field(default_factory=list)
    decode_counts: list[int] = field(default_factory=list)
    num_iterations: int = 0

    def save(self, filepath: str | Path) -> None:
        """Save parameters to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info('Hardware parameters saved to %s', filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> 'HardwareParams':
        """Load parameters from JSON file. Tolerates unknown extra keys."""
        with open(filepath) as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def __str__(self) -> str:
        return f'HardwareParams(\n  α_p={self.alpha_p:.6f}s, β_p={self.beta_p:.8f}s/token\n  α_d={self.alpha_d:.6f}s, β_d={self.beta_d:.8f}s/req\n  model={self.model}, device={self.device_name}\n  prefill_r2={self.prefill_r2:.4f}, decode_r2={self.decode_r2:.4f}\n)'

def load_hardware_params(filepath: str | Path | None=None) -> HardwareParams | None:
    """
    Load hardware parameters from file.

    Checks in order:
    1. Provided filepath
    2. VLLM_PD_CALIBRATION_FILE environment variable

    Returns None if no file is found.
    """
    if filepath is None:
        filepath = os.environ.get('VLLM_PD_CALIBRATION_FILE', '')
    if not filepath:
        return None
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning('Calibration file not found: %s', filepath)
        return None
    try:
        params = HardwareParams.load(filepath)
        logger.info('Loaded hardware parameters from %s', filepath)
        return params
    except Exception as e:
        logger.warning('Failed to load calibration file: %s', e)
        return None

class AuthorPolicy:

    def _initialize_author_policy(self):
        _mode_env = os.environ.get('VLLM_PD_SCHEDULER_MODE', '')
        if _mode_env:
            self.scheduler_mode = _mode_env.lower()
        elif os.environ.get('VLLM_USE_PD_SCHEDULER', '0') == '1':
            self.scheduler_mode = 'eb'
        else:
            self.scheduler_mode = 'v1'
        self.use_pd_scheduler = self.scheduler_mode in ('eb', 'auto')
        if self.use_pd_scheduler:
            self.pd_batch_size_N = self.max_num_running_reqs
            self.pd_k_mode = os.environ.get('VLLM_PD_K_MODE', 'direct')
            _k_star_env = os.environ.get('VLLM_PD_K_STAR', '')
            self.pd_k_star_user_specified = bool(_k_star_env)
            self.pd_k_star_fixed = int(_k_star_env) if _k_star_env else 0
            _k_ratio_env = os.environ.get('VLLM_PD_K_RATIO', '')
            self.pd_k_ratio_user_specified = bool(_k_ratio_env)
            self.pd_k_ratio = float(_k_ratio_env) if _k_ratio_env else 0.0
            if self.scheduler_mode == 'auto' and self.pd_k_mode == 'direct' and (not self.pd_k_ratio_user_specified) and (not self.pd_k_star_user_specified):
                logger.warning('VLLM_PD_SCHEDULER_MODE=auto with VLLM_PD_K_MODE=direct and no VLLM_PD_K_RATIO / VLLM_PD_K_STAR override: the MB↔EB crossover will never fire (pd_k_ratio stays 0) and auto will behave like cold-start mode. Set VLLM_PD_K_MODE=ifr (recommended) or cfr to enable online switching.')
            _hw_params = load_hardware_params()
            if _hw_params is not None:
                self.pd_alpha_p = _hw_params.alpha_p
                self.pd_beta_p = _hw_params.beta_p
                self.pd_alpha_d = _hw_params.alpha_d
                self.pd_beta_d = _hw_params.beta_d
                logger.info('Loaded hardware params from calibration file: model=%s, device=%s', _hw_params.model, _hw_params.device_name)
            else:
                _alpha_p = os.environ.get('VLLM_PD_ALPHA_P')
                _beta_p = os.environ.get('VLLM_PD_BETA_P')
                _alpha_d = os.environ.get('VLLM_PD_ALPHA_D')
                _beta_d = os.environ.get('VLLM_PD_BETA_D')
                if not all([_alpha_p, _beta_p, _alpha_d, _beta_d]):
                    raise ValueError('PD Scheduler requires hardware calibration parameters. Please run calibration first:\n  python -m vllm.v1.core.sched.calibration --model <model_name>\nOr set environment variable:\n  export VLLM_PD_CALIBRATION_FILE=/path/to/pd_calibration.json\nOr set all timing parameters manually:\n  export VLLM_PD_ALPHA_P=<value>\n  export VLLM_PD_BETA_P=<value>\n  export VLLM_PD_ALPHA_D=<value>\n  export VLLM_PD_BETA_D=<value>')
                assert _alpha_p is not None
                assert _beta_p is not None
                assert _alpha_d is not None
                assert _beta_d is not None
                self.pd_alpha_p = float(_alpha_p)
                self.pd_beta_p = float(_beta_p)
                self.pd_alpha_d = float(_alpha_d)
                self.pd_beta_d = float(_beta_d)
            self.pd_p = 0.01
            if self.pd_k_mode in ('ifr', 'cfr'):
                self.pd_ifr_window_size = int(os.environ.get('VLLM_PD_IFR_WINDOW_SIZE', '500'))
                self.pd_ifr_samples: deque[int] = deque(maxlen=self.pd_ifr_window_size)
                self.pd_ifr_update_interval = int(os.environ.get('VLLM_PD_IFR_UPDATE_INTERVAL', '100'))
                self.pd_ifr_min_samples = int(os.environ.get('VLLM_PD_IFR_MIN_SAMPLES', '30'))
                self.pd_ifr_default_theta = float(os.environ.get('VLLM_PD_IFR_DEFAULT_THETA', '0.70'))
                self.pd_ifr_update_counter = 0
                self.pd_hazard_p0 = 0.01
                self.pd_hazard_eta = 0.0
                self.pd_theta_max = float(os.environ.get('VLLM_PD_THETA_MAX', '0.80'))
                self.pd_theta_floor = float(os.environ.get('VLLM_PD_THETA_FLOOR', '0.01'))
                self.pd_ifr_theta_ema_alpha = float(os.environ.get('VLLM_PD_IFR_THETA_EMA_ALPHA', '0.3'))
                self.pd_ifr_theta_initialized = False
            if self.pd_k_mode == 'direct':
                if self.pd_k_star_user_specified:
                    self.pd_switch_threshold_k = max(1, self.pd_k_star_fixed)
                else:
                    self.pd_switch_threshold_k = self._compute_optimal_k()
            elif self.pd_k_mode == 'ratio':
                if not self.pd_k_ratio_user_specified:
                    self.pd_k_ratio = self._compute_optimal_ratio()
                self.pd_switch_threshold_k = self._compute_k_from_ratio()
            elif self.pd_k_mode == 'ifr':
                self.pd_k_ratio = self.pd_ifr_default_theta
                self.pd_switch_threshold_k = self._compute_k_from_ratio()
            elif self.pd_k_mode == 'cfr':
                self.pd_k_ratio = self.pd_ifr_default_theta
                self.pd_switch_threshold_k = self._compute_k_from_ratio()
                self.pd_cfr_initialized = False
            else:
                logger.warning("Unknown k mode '%s', using direct mode", self.pd_k_mode)
                self.pd_k_mode = 'direct'
                self.pd_switch_threshold_k = self._compute_optimal_k()
            if self.pd_k_mode == 'direct':
                dyn_tag = 'fixed' if self.pd_k_star_user_specified else 'auto'
                k_info = f'k*={self.pd_switch_threshold_k} ({dyn_tag})'
            elif self.pd_k_mode == 'ratio':
                dyn_tag = 'fixed' if self.pd_k_ratio_user_specified else 'auto'
                k_info = f'θ*={self.pd_k_ratio:.4f}, k*={self.pd_switch_threshold_k} ({dyn_tag})'
            elif self.pd_k_mode == 'cfr':
                k_info = f'θ*={self.pd_k_ratio:.4f}, k*={self.pd_switch_threshold_k} (CFR midpoint, θ_max={self.pd_theta_max})'
            else:
                k_info = f'θ*={self.pd_k_ratio:.4f}, k*={self.pd_switch_threshold_k} (IFR adaptive, θ_max={self.pd_theta_max})'
            logger.info(f'[P/D Competition Scheduler] Initialized: N={self.pd_batch_size_N}, k_mode={k_info}, α_p={self.pd_alpha_p}, β_p={self.pd_beta_p}, α_d={self.pd_alpha_d}, β_d={self.pd_beta_d}')
            self.pd_phase = 0
            self.pd_prefilled_count = 0
            self.pd_completed_decode_count = 0
            self.pd_refill_target = 0
            self.pd_decoding_requests: set[str] = set()
            self.pd_param_update_interval = int(os.environ.get('VLLM_PD_PARAM_UPDATE_INTERVAL', '100'))
            self.pd_ema_alpha = 0.2
            self.pd_total_completed = 0
            self.pd_param_initialized = False
            self.pd_batch_completed_count = 0
            self.pd_batch_total_output_tokens = 0
            self.pd_avg_output_tokens = 100.0
            self.pd_base_kv_reserve = float(os.environ.get('VLLM_PD_BASE_KV_RESERVE', '0'))
            self.pd_output_margin = float(os.environ.get('VLLM_PD_OUTPUT_MARGIN', '0.5'))
            self.pd_last_n_update_time = 0.0
            self.pd_n_update_cooldown = float(os.environ.get('VLLM_PD_N_UPDATE_COOLDOWN', '2.0'))
            self.pd_avg_prompt_len = 512.0
            self.pd_avg_prompt_ema_alpha = 0.05
            self.pd_auto_compute_n = os.environ.get('VLLM_PD_AUTO_COMPUTE_N', '0') == '1'
            self.pd_oom_tolerance = float(os.environ.get('VLLM_PD_OOM_TOLERANCE', '0.01'))
            self.pd_oom_event_count = 0
            self.pd_update_history: list[dict] = []
            self.pd_theta_zero_last = 0.0
            self.pd_k_hat_midpoint_last = 0.0
            self.pd_n_hat_safe_last = 0.0
            self.pd_delta_diagnostic_last = 0.0
            logger.info(f'[P/D Adaptive] Initial: avg_output={self.pd_avg_output_tokens:.0f}, base_kv_reserve={self.pd_base_kv_reserve:.2f}, output_margin={self.pd_output_margin:.1f}, auto_compute_N={self.pd_auto_compute_n}, OOM_tol_eps={self.pd_oom_tolerance}')
        else:
            logger.info('[Scheduler] Using original vLLM scheduler')
        if self.scheduler_mode == 'auto':
            self._active_scheduler = os.environ.get('VLLM_PD_AUTO_COLD_START_MODE', 'mb')
            self._mb_cost_a = float(os.environ.get('VLLM_PD_MB_COST_A', '0'))
            self._mb_cost_b = float(os.environ.get('VLLM_PD_MB_COST_B', '0'))
            self._mb_cost_c = float(os.environ.get('VLLM_PD_MB_COST_C', '0'))
            self._mb_cost_profiled = any((os.environ.get(k) for k in ['VLLM_PD_MB_COST_A', 'VLLM_PD_MB_COST_B', 'VLLM_PD_MB_COST_C']))
            _acp = os.environ.get('VLLM_PD_ALPHA_MB', '')
            self._alpha_mb = float(_acp) if _acp else self.pd_alpha_p
            self._mode_switch_delta = float(os.environ.get('VLLM_PD_MODE_SWITCH_DELTA', '0.0001'))
            self._mode_cooldown_max = int(os.environ.get('VLLM_PD_MODE_COOLDOWN', '3'))
            self._mode_cooldown = 0
            self._n_obs = float(self.max_num_running_reqs)
            self._avg_prompt_len = 512.0
            self._avg_prompt_ema_alpha = 0.05
            self._mode_switch_history: list[dict] = []
            self._mode_switch_count = 0
            logger.info(f'[EB+] Auto mode initialized: cold_start={self._active_scheduler}, mb_cost_profiled={self._mb_cost_profiled}, alpha_mb={self._alpha_mb:.6f}, delta={self._mode_switch_delta}, cooldown={self._mode_cooldown_max}')
        self.chunk_prefilling: list[Request] = []
        self.pd_n_update_history: list[dict] = []
        self._pd_start_time = time.monotonic()
        self._param_update_count = 0
        self._param_update_total_us = 0.0
        self._last_param_update_us = 0.0
        self._schedule_stats_enabled = os.environ.get('VLLM_COLLECT_SCHEDULE_STATS', '0') == '1'
        self._schedule_stats: list[dict] = []
        self._schedule_stats_start_time: float | None = None
        self._schedule_stats_file = os.environ.get('VLLM_SCHEDULE_STATS_FILE', 'schedule_stats.json')
        if self._schedule_stats_enabled:
            atexit.register(self._save_stats_on_exit)
    PD_PHASE_NAMES = {0: 'INITIAL_PREFILL', 1: 'DECODE', 2: 'REFILL_PREFILL'}

    def get_pd_stats(self) -> dict:
        """Get current P/D scheduling statistics for monitoring."""
        stats = {'phase': self.pd_phase, 'k_star': self.pd_switch_threshold_k, 'k_ratio': self.pd_k_ratio, 'k_ratio_user_specified': self.pd_k_ratio_user_specified, 'k_mode': self.pd_k_mode, 'N': self.pd_batch_size_N, 'prefilled_count': self.pd_prefilled_count, 'completed_decode_count': self.pd_completed_decode_count, 'refill_target': self.pd_refill_target, 'decoding_requests': len(self.pd_decoding_requests), 'running_requests': len(self.running), 'waiting_requests': len(self.waiting), 'p': self.pd_p, 'total_completed': self.pd_total_completed, 'avg_output_tokens': self.pd_avg_output_tokens, 'adaptive_kv_threshold': self._compute_adaptive_kv_threshold(), 'adaptive_N': self._compute_adaptive_N()}
        if self.pd_k_mode == 'ifr':
            stats.update({'hazard_p0': self.pd_hazard_p0, 'hazard_eta': self.pd_hazard_eta, 'ifr_sample_count': len(self.pd_ifr_samples), 'ifr_update_counter': self.pd_ifr_update_counter, 'ifr_update_interval': self.pd_ifr_update_interval, 'ifr_window_size': self.pd_ifr_window_size, 'theta_max': self.pd_theta_max})
        return stats

    def _compute_optimal_k(self) -> int:
        """
        Compute optimal switching threshold k* using Proposition 1.

        k* is the smallest integer k satisfying:
            k * τ(N-k) - Σ_{j=N-k+1}^{N} τ(j) >= α_p

        This maximizes throughput = k / (E[T_d(k)] + E[T_p(k)])

        Optimized: O(N) instead of O(N²) via τ precomputation + incremental sum.
        """
        N = self.pd_batch_size_N
        one_minus_p = 1.0 - self.pd_p
        tau = []
        power = 1.0
        for j in range(N + 1):
            if j == 0:
                tau.append(float('inf'))
            else:
                power *= one_minus_p
                denom = 1.0 - power
                if denom <= 0:
                    tau.append(float('inf'))
                else:
                    tau.append((self.pd_alpha_d + self.pd_beta_d * j) / denom)
        sum_tau = 0.0
        for k in range(1, N + 1):
            sum_tau += tau[N - k + 1]
            lhs = k * tau[N - k]
            rhs = sum_tau + self.pd_alpha_p
            if lhs >= rhs:
                return max(1, k)
        return max(1, N // 5)

    def _compute_k_from_ratio(self) -> int:
        """
        Compute k* as a ratio of N.

        k* = pd_k_ratio * N

        This makes k* automatically adapt when N changes (e.g., due to
        adaptive N learning based on avg output tokens).

        Returns:
            int: k* value (at least 1)
        """
        k = int(self.pd_k_ratio * self.pd_batch_size_N)
        return max(1, k)

    def _compute_optimal_ratio(self) -> float:
        """
        Compute optimal ratio θ* using asymptotic formula (Proposition 1).

        For long sequences (p << 1) and moderate batch sizes (N << 1/p),
        the normalized threshold θ* = k*/N satisfies:

            θ/(1-θ) + ln(1-θ) = p * α_p / α_d

        This is solved using bisection method.

        Returns:
            float: optimal ratio θ* in (0, 1)
        """
        import math
        C = self.pd_p * self.pd_alpha_p / self.pd_alpha_d

        def f(theta: float) -> float:
            if theta <= 0 or theta >= 1:
                return float('inf')
            return theta / (1 - theta) + math.log(1 - theta)
        if C <= 1e-06:
            return _PD_THETA_HARD_MIN
        lo, hi = (0.001, 0.999)
        for _ in range(100):
            mid = (lo + hi) / 2
            f_mid = f(mid)
            if abs(f_mid - C) < 1e-09:
                break
            if f_mid < C:
                lo = mid
            else:
                hi = mid
        theta_star = (lo + hi) / 2
        theta_star = max(_PD_THETA_HARD_MIN, min(0.99, theta_star))
        logger.debug(f'Computed optimal ratio: θ*={theta_star:.4f} (p={self.pd_p}, α_p={self.pd_alpha_p}, α_d={self.pd_alpha_d}, C={C:.6f})')
        return theta_star

    def _compute_theta_zero_exact(self, p_0: float) -> float:
        """θ_0 from the exact CFR formula (Eq. theta_base):

            θ/(1-θ) + ln(1-θ) = (-ln(1-p_0)) · α_p / α_d.

        Differs from `_compute_optimal_ratio` (which uses p_0 directly on the
        right-hand side, valid only as p_0 → 0) by the exact -ln(1-p_0)
        factor required by the paper's CFR proofs.
        """
        import math
        if p_0 <= 0.0:
            return _PD_THETA_HARD_MIN
        if p_0 >= 1.0 - 1e-09:
            return 0.99
        rhs = -math.log(1.0 - p_0) * self.pd_alpha_p / self.pd_alpha_d
        if rhs <= 1e-12:
            return _PD_THETA_HARD_MIN

        def f(theta: float) -> float:
            if theta <= 0.0 or theta >= 1.0:
                return float('inf')
            return theta / (1.0 - theta) + math.log(1.0 - theta)
        lo, hi = (1e-06, 1.0 - 1e-06)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(mid) < rhs:
                lo = mid
            else:
                hi = mid
        theta_zero = 0.5 * (lo + hi)
        return max(_PD_THETA_HARD_MIN, min(0.99, theta_zero))

    def _compute_midpoint_k(self, theta_zero: float, p_0: float, n: int, mu_L: float) -> tuple[float, int, int, int]:
        """Midpoint k̂ following the closed-form midpoint construction.

        The R̄(m) / TP_int(m) / k̂ definitions used below are introduced
        in the journal version of the paper, not the camera-ready
        proceedings paper.

        Returns
        -------
        (k_hat, M_minus, M_plus, M_star) where k_hat is the (real-valued)
        midpoint placement; the integer threshold actually used is
        max(1, round(k_hat)).
        """
        import math
        if p_0 <= 0.0 or p_0 >= 1.0 or theta_zero <= 0.0 or (theta_zero >= 1.0):
            return (float(max(1, int(theta_zero * n))), 0, 0, 0)
        if n <= 1:
            return (1.0, 0, 0, 0)
        log_one_minus_theta = math.log(1.0 - theta_zero)
        log_one_minus_p = math.log(1.0 - p_0)
        if log_one_minus_p == 0.0:
            return (float(max(1, int(theta_zero * n))), 0, 0, 0)
        tau_real = log_one_minus_theta / log_one_minus_p
        m_minus = max(1, int(math.floor(tau_real)))
        m_plus = max(m_minus + 1, int(math.ceil(tau_real)))

        def r_bar(m: int) -> float:
            if m <= 0:
                return 0.0
            return n * (1.0 - (1.0 - p_0) ** m)

        def tp_int(m: int) -> float:
            r = r_bar(m)
            denom = self.pd_alpha_d * m + self.pd_beta_d * r / max(p_0, 1e-09) + self.pd_alpha_p + self.pd_beta_p * r * mu_L
            return r / denom if denom > 0 else 0.0
        m_star = m_minus if tp_int(m_minus) >= tp_int(m_plus) else m_plus
        k_hat = 0.5 * (r_bar(max(0, m_star - 1)) + r_bar(m_star))
        k_hat = max(1.0, min(float(n - 1), k_hat))
        return (k_hat, m_minus, m_plus, m_star)

    def _compute_memory_safe_n(self, theta_zero: float, p_0: float, mu_L: float) -> int:
        """Memory-safe N̂ from Proposition prop:memory.

        NOTE on paper-vs-code:
        The main paper (Eq. eq:Nstar) presents the supremum bound
            x_ε = ν · ln(1/ε),    ν = 1/(p_0² · μ_L)
        giving the linearised closed form
            N* = ⌊ (C - ν · ln(1/ε)) / D(θ) ⌋.
        This implementation instead solves the CLT-type concentration
            N · D(θ) + σ(θ) · sqrt(N · ln(1/ε)) ≤ C
        for N (returning the floor of the positive root, clamped to
        [1, max_num_running_reqs]). The two bounds are asymptotically
        equivalent for N ≫ 1; the CLT form is tighter in the moderate-N
        regime relevant to our experiments, and is the form actually used
        to produce all paper results. The journal version of the paper
        adopts this quadratic form.

        Here:
            D(θ) = μ_L + (1-θ)/(θ p_0) · Λ        with Λ = -ln(1-θ)
            σ²(θ) = 2Λ (1 + (p_0 μ_L + Λ/θ)²) / p_0²
        and ε = `pd_oom_tolerance`.
        """
        import math
        n_cap = int(self.max_num_running_reqs)
        if not self.pd_auto_compute_n:
            return n_cap
        try:
            num_blocks = int(self.kv_cache_manager.block_pool.num_gpu_blocks)
            block_size = int(self.block_size)
            capacity_tokens = num_blocks * block_size
        except Exception:
            return n_cap
        if capacity_tokens <= 0:
            return n_cap
        eps = max(min(self.pd_oom_tolerance, 0.5), 1e-06)
        log_inv_eps = math.log(1.0 / eps)
        if theta_zero <= 0.0 or theta_zero >= 1.0 or p_0 <= 0.0:
            return n_cap
        Lambda = -math.log(1.0 - theta_zero)
        D_theta = mu_L + (1.0 - theta_zero) / (theta_zero * p_0) * Lambda
        if D_theta <= 0:
            return n_cap
        lambda_theta = Lambda / max(theta_zero, 1e-09)
        sigma_sq = 2.0 * Lambda * (1.0 + (p_0 * mu_L + lambda_theta) ** 2) / max(p_0 ** 2, 1e-18)
        sigma = math.sqrt(max(sigma_sq, 0.0))
        sqrt_N = (-sigma * math.sqrt(log_inv_eps) + math.sqrt(sigma_sq * log_inv_eps + 4.0 * D_theta * capacity_tokens)) / (2.0 * D_theta)
        n_safe = int(math.floor(max(0.0, sqrt_N) ** 2))
        return max(1, min(n_cap, n_safe))

    def _compute_diagnostic_delta(self, theta_zero: float, k_hat: float, n: int, mu_L: float, mu_O: float) -> float:
        """Diagnostic Δ(N) for the adaptive selector.

        Equivalent to the inequality in Eq. eq:comparison_condition
        (Prop. 4, prop:comparison) rearranged as LHS - RHS:
        Δ(N) < 0  iff  MB throughput > EB throughput.

        Δ(N) = (β_MB^e − β_EB^w)
             − (1/(μ_L+μ_O)) · [ (α_p − α_d ln(1−θ_0) μ_O)/k̂
                               − α_MB(1+μ_O)/N ]

        The kernel-cost terms (β_MB^e, α_MB) are read from environment-
        provided calibration constants (VLLM_PD_BETA_MB_E, VLLM_PD_ALPHA_MB)
        — defaulting to 0 when not profiled (in which case Δ degenerates
        but the sign of the second term still drives MB/EB choice).
        """
        import math
        if mu_L + mu_O <= 0 or k_hat <= 0 or n <= 0:
            return 0.0
        beta_eb_w = (self.pd_beta_p * mu_L + self.pd_beta_d * mu_O) / (mu_L + mu_O)
        beta_mb_e = float(os.environ.get('VLLM_PD_BETA_MB_E', str(beta_eb_w)))
        alpha_mb = float(os.environ.get('VLLM_PD_ALPHA_MB', str(self.pd_alpha_p)))
        log_one_minus_theta = math.log(max(1.0 - theta_zero, 1e-09))
        eb_term = (self.pd_alpha_p - self.pd_alpha_d * log_one_minus_theta * mu_O) / k_hat
        mb_term = alpha_mb * (1.0 + mu_O) / n
        return beta_mb_e - beta_eb_w - (eb_term - mb_term) / (mu_L + mu_O)

    def _estimate_hazard_params(self) -> tuple[float, float]:
        """
        Estimate hazard rate parameters (p_0, η) from sliding window samples.

        The empirical hazard rate at iteration t is:
            ĥ(t) = #{O_i = t} / #{O_i >= t}

        We fit h(t) = p_0 + η * t via weighted least squares over
        t ∈ [t_start, t_95], where t_start is the 5th percentile of
        observed output lengths.  Fitting only over the support of the
        distribution avoids the zero-hazard prefix that arises with
        bounded-support distributions (e.g. uniform, gamma with large
        shape), which would otherwise drag p_0 negative.

        Returns:
            tuple[float, float]: (p_0, η) where η >= 0 for IFR distributions
        """
        samples = self.pd_ifr_samples
        if len(samples) < self.pd_ifr_min_samples:
            return (self.pd_hazard_p0, self.pd_hazard_eta)
        sorted_samples = sorted(samples)
        t_start = sorted_samples[max(0, int(len(sorted_samples) * 0.05))]
        t_start = max(t_start, 1)
        t_95 = sorted_samples[int(len(sorted_samples) * 0.95)]
        t_95 = max(t_95, t_start + 10)
        from collections import Counter
        counts = Counter(samples)
        max_t = max(samples)
        survivors = [0] * (max_t + 2)
        survivors[max_t + 1] = 0
        for t in range(max_t, 0, -1):
            survivors[t] = survivors[t + 1] + counts.get(t, 0)
        sum_w = 0.0
        sum_wt = 0.0
        sum_wt2 = 0.0
        sum_wh = 0.0
        sum_wth = 0.0
        for t in range(t_start, min(t_95 + 1, max_t + 1)):
            n_t = survivors[t]
            if n_t < 5:
                continue
            d_t = counts.get(t, 0)
            h_t = d_t / n_t
            w = n_t
            sum_w += w
            sum_wt += w * t
            sum_wt2 += w * t * t
            sum_wh += w * h_t
            sum_wth += w * t * h_t
        if sum_w < 10:
            return (self.pd_hazard_p0, self.pd_hazard_eta)
        det = sum_w * sum_wt2 - sum_wt * sum_wt
        if abs(det) < 1e-10:
            sample_mean = sum(samples) / len(samples)
            p_0 = 1.0 / sample_mean if sample_mean > 0 else 0.01
            return (p_0, 0.0)
        p_0 = (sum_wt2 * sum_wh - sum_wt * sum_wth) / det
        eta = (sum_w * sum_wth - sum_wt * sum_wh) / det
        eta = max(0.0, eta)
        sample_mean = sum(samples) / len(samples)
        p_0_floor = 1.0 / sample_mean if sample_mean > 0 else 0.01
        p_0 = max(p_0, p_0_floor)
        return (p_0, eta)

    def _compute_ifr_correction(self, theta_cfr: float) -> float:
        """
        Compute IFR correction Δθ based on the IFR threshold theorem.

        See `thm:threshold_ifr` in the paper.

        For linear increasing hazard rate h(t) = p_0 + η * t with η > 0,
        the optimal threshold admits:
            θ*_IFR = θ*_CFR + Δθ

        where:
            Δθ = (η(1-θ*_CFR)²) / (p_0² * θ*_CFR) *
                 [Λ(θ*_CFR/(1-θ*_CFR) - Λ/2) + ρ(Λ - θ*_CFR)]

        with Λ = -ln(1-θ*_CFR) and ρ = β_d * N / α_d.

        Args:
            theta_cfr: The CFR baseline threshold θ*_CFR

        Returns:
            float: The correction Δθ (always >= 0)
        """
        if self.pd_hazard_eta <= 0 or theta_cfr <= 0 or theta_cfr >= 1:
            return 0.0
        import math
        p_0 = self.pd_hazard_p0
        eta = self.pd_hazard_eta
        Lambda = -math.log(1 - theta_cfr)
        rho = self.pd_beta_d * self.pd_batch_size_N / self.pd_alpha_d
        duration_effect = Lambda * (theta_cfr / (1 - theta_cfr) - Lambda / 2)
        per_token_effect = rho * (Lambda - theta_cfr)
        numerator = eta * (1 - theta_cfr) ** 2
        denominator = p_0 ** 2 * theta_cfr
        if denominator < 1e-12:
            return 0.0
        delta_theta = numerator / denominator * (duration_effect + per_token_effect)
        delta_theta = max(0.0, delta_theta)
        delta_theta = min(delta_theta, 5.0 * theta_cfr)
        return delta_theta

    def _compute_optimal_ratio_ifr(self) -> float:
        """
        Compute optimal ratio θ* with IFR correction.

        This implements the online adaptive threshold selection
        (`alg:adaptive_joint` in the paper):
        1. Estimate hazard rate parameters (p_0, η) from samples
        2. Compute CFR baseline θ*_CFR using Proposition 1
        3. If η > 0, apply IFR correction from the IFR threshold
           theorem (`thm:threshold_ifr`)
        4. Return θ* = min(θ*_CFR + Δθ, θ_max)

        Returns:
            float: Optimal ratio θ* in (0, θ_max]
        """
        p_0, eta = self._estimate_hazard_params()
        self.pd_hazard_p0 = p_0
        self.pd_hazard_eta = eta
        old_p = self.pd_p
        self.pd_p = p_0
        theta_cfr = self._compute_optimal_ratio()
        self.pd_p = old_p
        if eta > 0:
            delta_theta = self._compute_ifr_correction(theta_cfr)
            theta_star = theta_cfr + delta_theta
        else:
            theta_star = theta_cfr
        theta_star = min(theta_star, self.pd_theta_max)
        theta_star = max(_PD_THETA_HARD_MIN, theta_star)
        logger.debug(f'IFR optimal ratio: θ*={theta_star:.4f} (θ*_CFR={theta_cfr:.4f}, Δθ={theta_star - theta_cfr:.4f}, p_0={p_0:.6f}, η={eta:.8f})')
        return theta_star

    def _update_ifr_threshold(self) -> None:
        """
        Online joint adaptation of (k̂*, N̂*) — paper Algorithm 1
        (alg:adaptive_joint).

        Called every M completions when window has >= W_min samples.
        Estimates (p̂_0, η̂, μ̂_L), computes θ_0 + Δθ → θ̂*, applies the
        memory-safe N̂* (Eq. eq:Nstar) when pd_auto_compute_n is enabled,
        and finally sets k̂* = ⌊θ̂* · N̂*⌋.

        EMA smoothing is applied to θ̂* to damp oscillations caused by
        noisy hazard-rate estimates.
        """
        old_ratio = self.pd_k_ratio
        old_k = self.pd_switch_threshold_k
        old_n = self.pd_batch_size_N
        p_0, eta = self._estimate_hazard_params()
        self.pd_hazard_p0 = p_0
        self.pd_hazard_eta = eta
        old_p = self.pd_p
        self.pd_p = p_0
        theta_0 = self._compute_optimal_ratio()
        self.pd_p = old_p
        if eta > 0:
            delta_theta = self._compute_ifr_correction(theta_0)
            theta_star = theta_0 + delta_theta
        else:
            delta_theta = 0.0
            theta_star = theta_0
        theta_star = max(_PD_THETA_HARD_MIN, self.pd_theta_floor, min(theta_star, self.pd_theta_max))
        if not self.pd_ifr_theta_initialized:
            self.pd_k_ratio = theta_star
            self.pd_ifr_theta_initialized = True
        else:
            alpha = self.pd_ifr_theta_ema_alpha
            self.pd_k_ratio = alpha * theta_star + (1 - alpha) * self.pd_k_ratio
        mu_L = max(1.0, float(self.pd_avg_prompt_len))
        mu_O = max(1.0, float(self.pd_avg_output_tokens))
        if self.pd_auto_compute_n:
            n_hat = self._compute_memory_safe_n(self.pd_k_ratio, p_0, mu_L)
            if n_hat != self.pd_batch_size_N:
                self._record_n_update(self.pd_batch_size_N, n_hat, 'ifr_memory_safe')
                self.pd_batch_size_N = n_hat
        n_eff = max(1, self.pd_batch_size_N)
        self.pd_switch_threshold_k = max(1, int(self.pd_k_ratio * n_eff))
        delta_diag = self._compute_diagnostic_delta(theta_0, float(self.pd_switch_threshold_k), n_eff, mu_L, mu_O)
        self.pd_theta_zero_last = theta_0
        self.pd_k_hat_midpoint_last = float(self.pd_switch_threshold_k)
        self.pd_n_hat_safe_last = float(n_eff)
        self.pd_delta_diagnostic_last = delta_diag
        self.pd_update_history.append({'timestamp': time.monotonic() - self._pd_start_time, 'p_0_estimate': p_0, 'eta_estimate': eta, 'mu_L_estimate': mu_L, 'mu_O_estimate': mu_O, 'theta_0': theta_0, 'delta_theta': delta_theta, 'theta_star': self.pd_k_ratio, 'k_hat_int': int(self.pd_switch_threshold_k), 'N_hat': int(n_eff), 'delta_diagnostic': delta_diag, 'samples_used': len(self.pd_ifr_samples), 'oom_event_count': int(self.pd_oom_event_count)})
        if abs(self.pd_k_ratio - old_ratio) > 0.01 or old_k != self.pd_switch_threshold_k or old_n != self.pd_batch_size_N:
            logger.info(f'[P/D IFR] online update: θ_0={theta_0:.4f}, θ̂={old_ratio:.4f}->{self.pd_k_ratio:.4f} (Δθ={delta_theta:.4f}), k̂={old_k}->{self.pd_switch_threshold_k}, N̂={old_n}->{self.pd_batch_size_N} (p_0={p_0:.6f}, η={eta:.8f}, samples={len(self.pd_ifr_samples)}, Δ={delta_diag:.6f})')

    def _update_cfr_threshold(self) -> None:
        """Online CFR midpoint update (closed-form midpoint construction
        not in the camera-ready paper; see the journal version).

        Estimates p_0 from the sliding window, recomputes the exact θ_0
        (Eq. eq:theta_base), the memory-safe N̂ (Eq. eq:Nstar) when
        enabled, and the midpoint k̂ (journal-only formula).  Also
        evaluates the diagnostic Δ(N) used by the adaptive selector.
        """
        old_ratio = self.pd_k_ratio
        old_k = self.pd_switch_threshold_k
        old_n = self.pd_batch_size_N
        p_0, _eta = self._estimate_hazard_params()
        self.pd_hazard_p0 = p_0
        self.pd_hazard_eta = 0.0
        mu_L = max(1.0, float(self.pd_avg_prompt_len))
        mu_O = max(1.0, float(self.pd_avg_output_tokens))
        theta_zero = self._compute_theta_zero_exact(p_0)
        theta_zero_clamped = max(_PD_THETA_HARD_MIN, min(self.pd_theta_max, theta_zero))
        if self.pd_auto_compute_n:
            n_hat = self._compute_memory_safe_n(theta_zero_clamped, p_0, mu_L)
            if n_hat != self.pd_batch_size_N:
                self._record_n_update(self.pd_batch_size_N, n_hat, 'cfr_memory_safe')
                self.pd_batch_size_N = n_hat
        n_eff = max(1, self.pd_batch_size_N)
        k_hat_real, m_minus, m_plus, m_star = self._compute_midpoint_k(theta_zero_clamped, p_0, n_eff, mu_L)
        delta_diag = self._compute_diagnostic_delta(theta_zero_clamped, k_hat_real, n_eff, mu_L, mu_O)
        theta_hat = k_hat_real / float(n_eff)
        theta_hat = max(_PD_THETA_HARD_MIN, min(self.pd_theta_max, theta_hat))
        if not self.pd_ifr_theta_initialized:
            self.pd_k_ratio = theta_hat
            self.pd_ifr_theta_initialized = True
        else:
            alpha = self.pd_ifr_theta_ema_alpha
            self.pd_k_ratio = alpha * theta_hat + (1 - alpha) * self.pd_k_ratio
        self.pd_switch_threshold_k = max(1, int(self.pd_k_ratio * n_eff))
        self.pd_theta_zero_last = theta_zero_clamped
        self.pd_k_hat_midpoint_last = k_hat_real
        self.pd_n_hat_safe_last = float(n_eff)
        self.pd_delta_diagnostic_last = delta_diag
        self.pd_update_history.append({'timestamp': time.monotonic() - self._pd_start_time, 'p_0_estimate': p_0, 'mu_L_estimate': mu_L, 'mu_O_estimate': mu_O, 'theta_0': theta_zero_clamped, 'k_hat_real': k_hat_real, 'k_hat_int': int(self.pd_switch_threshold_k), 'N_hat': int(n_eff), 'M_minus': int(m_minus), 'M_plus': int(m_plus), 'M_star': int(m_star), 'delta_diagnostic': delta_diag, 'samples_used': len(self.pd_ifr_samples), 'oom_event_count': int(self.pd_oom_event_count)})
        if abs(self.pd_k_ratio - old_ratio) > 0.01 or old_k != self.pd_switch_threshold_k or old_n != self.pd_batch_size_N:
            logger.info(f'[P/D CFR] midpoint update: θ_0={theta_zero_clamped:.4f}, θ̂={old_ratio:.4f}->{self.pd_k_ratio:.4f}, k̂={old_k}->{self.pd_switch_threshold_k}, N̂={old_n}->{self.pd_batch_size_N} (p_0={p_0:.6f}, μ_L={mu_L:.0f}, μ_O={mu_O:.0f}, M*={m_star}, Δ={delta_diag:.6f})')

    def _record_n_update(self, old_n: int, new_n: int, reason: str) -> None:
        """Record an N update event for trajectory tracking."""
        if old_n == new_n:
            return
        timestamp = time.monotonic() - self._pd_start_time
        self.pd_n_update_history.append({'timestamp': timestamp, 'old_N': old_n, 'new_N': new_n, 'reason': reason, 'k_star': self.pd_switch_threshold_k, 'avg_output_tokens': self.pd_avg_output_tokens})

    def _compute_adaptive_kv_threshold(self) -> float:
        """
        Compute adaptive KV cache threshold based on average output tokens.

        The idea: Reserve enough KV cache space for decoding phase.
        - If avg_output_tokens is high, reserve more space (higher threshold)
        - If avg_output_tokens is low, can use more cache for prefill

        Formula:
        - expected_decode_blocks = N * avg_output_tokens * margin / block_size
        - threshold = expected_decode_blocks / total_blocks + base_reserve

        Returns:
            float: KV cache threshold (fraction of total blocks to reserve)
        """
        if not hasattr(self.kv_cache_manager, 'block_pool'):
            return 0.05
        total_blocks = self.kv_cache_manager.block_pool.num_gpu_blocks
        if total_blocks <= 0:
            return 0.05
        tokens_per_block = self.block_size
        expected_output_tokens = self.pd_avg_output_tokens * self.pd_output_margin
        blocks_for_decode = self.pd_batch_size_N * expected_output_tokens / tokens_per_block
        reserve_ratio = blocks_for_decode / total_blocks
        kv_threshold_max = float(os.environ.get('VLLM_PD_KV_THRESHOLD_MAX', '0.6'))
        threshold = reserve_ratio + self.pd_base_kv_reserve
        threshold = max(0.05, min(kv_threshold_max, threshold))
        return threshold

    def _compute_adaptive_N(self) -> int:
        """
        Compute adaptive batch size N based on KV cache capacity and avg output.

        The idea: N should be chosen such that:
        - All N requests can be prefilled
        - There's enough KV cache left for decode phase

        Constraint:
        - N * (avg_prompt + avg_output * margin) / block_size
            <= total_blocks * (1 - reserve)

        Solving for N:
        - N <= total_blocks * (1 - reserve) * block_size
                / (avg_prompt + avg_output * margin)

        Returns:
            int: Adaptive batch size N
        """
        if not hasattr(self.kv_cache_manager, 'block_pool'):
            return self.max_num_running_reqs
        total_blocks = self.kv_cache_manager.block_pool.num_gpu_blocks
        if total_blocks <= 0:
            return self.max_num_running_reqs
        avg_prompt_tokens = 512
        sample_requests = list(self.running)[:50] + list(self.waiting)[:50]
        if sample_requests:
            total_prompt = sum((r.num_prompt_tokens for r in sample_requests))
            avg_prompt_tokens = total_prompt / len(sample_requests)
        expected_output = self.pd_avg_output_tokens * self.pd_output_margin
        tokens_per_request = avg_prompt_tokens + expected_output
        available_blocks = total_blocks * (1.0 - self.pd_base_kv_reserve)
        blocks_per_request = tokens_per_request / self.block_size
        adaptive_n = int(available_blocks / blocks_per_request)
        min_n = max(16, self.max_num_running_reqs // 10)
        max_n = self.max_num_running_reqs
        adaptive_n = max(min_n, min(max_n, adaptive_n))
        return adaptive_n

    def _update_params_online(self, output_tokens: int) -> None:
        """
        Unified parameter update with configurable interval.

        Updates all parameters together: avg_output_tokens, p, k*, θ*
        Hot path (every request): Only two integer additions.
        Cold path (every pd_param_update_interval requests): Update all params.

        k* update behavior by mode:
        - "direct": k* computed via Proposition 1
                    (unless VLLM_PD_K_STAR is user-specified)
        - "ratio":  k* = theta* x N, theta* computed from p
                    (unless VLLM_PD_K_RATIO is user-specified)
        - "ifr":    k* = theta* x N, theta* from the IFR correction formula
                    (using the hazard-rate estimate)
        """
        self.pd_batch_completed_count += 1
        self.pd_batch_total_output_tokens += output_tokens
        if self.pd_k_mode == 'ifr':
            self.pd_ifr_samples.append(output_tokens)
            self.pd_ifr_update_counter += 1
            if self.pd_ifr_update_counter >= self.pd_ifr_update_interval and len(self.pd_ifr_samples) >= self.pd_ifr_min_samples:
                self._update_ifr_threshold()
                self.pd_ifr_update_counter = 0
        elif self.pd_k_mode == 'cfr':
            self.pd_ifr_samples.append(output_tokens)
            self.pd_ifr_update_counter += 1
            if self.pd_ifr_update_counter >= self.pd_ifr_update_interval and len(self.pd_ifr_samples) >= self.pd_ifr_min_samples:
                self._update_cfr_threshold()
                self.pd_ifr_update_counter = 0
        if self.pd_batch_completed_count < self.pd_param_update_interval:
            return
        _cold_path_start = time.perf_counter()
        self.pd_total_completed += self.pd_batch_completed_count
        if self.pd_batch_total_output_tokens > 0:
            batch_mean_len = self.pd_batch_total_output_tokens / self.pd_batch_completed_count
            batch_p = 1.0 / batch_mean_len
            if not self.pd_param_initialized:
                self.pd_p = batch_p
                self.pd_avg_output_tokens = batch_mean_len
                self.pd_param_initialized = True
            else:
                self.pd_p = self.pd_ema_alpha * batch_p + (1 - self.pd_ema_alpha) * self.pd_p
                self.pd_avg_output_tokens = self.pd_ema_alpha * batch_mean_len + (1 - self.pd_ema_alpha) * self.pd_avg_output_tokens
            if self.pd_k_mode == 'direct' and (not self.pd_k_star_user_specified):
                old_k = self.pd_switch_threshold_k
                self.pd_switch_threshold_k = self._compute_optimal_k()
                if old_k != self.pd_switch_threshold_k:
                    logger.info(f'[P/D] k* update: {old_k}->{self.pd_switch_threshold_k} (p={self.pd_p:.4f}, mean_len={batch_mean_len:.1f})')
            elif self.pd_k_mode == 'ratio' and (not self.pd_k_ratio_user_specified):
                old_ratio = self.pd_k_ratio
                old_k = self.pd_switch_threshold_k
                self.pd_k_ratio = self._compute_optimal_ratio()
                if self.pd_k_ratio != old_ratio:
                    self.pd_switch_threshold_k = self._compute_k_from_ratio()
                    logger.info(f'[P/D] ratio update: θ*={old_ratio:.4f}->{self.pd_k_ratio:.4f}, k*={old_k}->{self.pd_switch_threshold_k} (p={self.pd_p:.4f}, mean_len={batch_mean_len:.1f})')
        if self.scheduler_mode == 'auto':
            self._evaluate_mode_switch()
        self.pd_batch_completed_count = 0
        self.pd_batch_total_output_tokens = 0
        _cold_path_elapsed = (time.perf_counter() - _cold_path_start) * 1000000.0
        self._param_update_count += 1
        self._param_update_total_us += _cold_path_elapsed
        self._last_param_update_us = _cold_path_elapsed

    def _evaluate_mode_switch(self) -> None:
        """Evaluate the EB-MB crossover condition (prop:comparison) and switch mode.

        Called every pd_param_update_interval completions from the cold path.
        Decision:
          LHS = β_MB^e(r̂) - β_EB_w
          RHS = (1/(μ_L+μ_o)) * [
                  (α_p - α_d·ln(1-θ₀)·μ_o)/(θ₀·N_obs)
                  - α_MB·(1+μ_o)/N_obs
                ]
          Switch to EB if LHS > RHS + δ  (contention dominates)
          Switch to CP if LHS < RHS - δ  (amortization dominates)
        """
        import math
        if not self.pd_param_initialized:
            return
        mu_o = self.pd_avg_output_tokens
        mu_L = self._avg_prompt_len
        if mu_L + mu_o <= 0:
            return
        r = mu_o / (mu_L + mu_o)
        N_obs = max(1.0, self._n_obs)
        beta_EB_w = (self.pd_beta_p * mu_L + self.pd_beta_d * mu_o) / (mu_L + mu_o)
        if self._mb_cost_profiled:
            beta_MB_e = self._mb_cost_a + self._mb_cost_b * r + self._mb_cost_c * r * r
        else:
            beta_MB_e = beta_EB_w
        LHS = beta_MB_e - beta_EB_w
        theta0 = self.pd_k_ratio if hasattr(self, 'pd_k_ratio') else 0.5
        if theta0 <= 0 or theta0 >= 1:
            return
        log_1_minus_theta = math.log(1 - theta0)
        eb_overhead = (self.pd_alpha_p - self.pd_alpha_d * log_1_minus_theta * mu_o) / (theta0 * N_obs)
        mb_overhead = self._alpha_mb * (1 + mu_o) / N_obs
        RHS = 1.0 / (mu_L + mu_o) * (eb_overhead - mb_overhead)
        delta = self._mode_switch_delta
        if self._mode_cooldown > 0:
            self._mode_cooldown -= 1
            return
        old_mode = self._active_scheduler
        if RHS + delta < LHS:
            if self._active_scheduler != 'eb':
                self._transition_to_eb()
                self._mode_cooldown = self._mode_cooldown_max
        elif RHS - delta > LHS and self._active_scheduler != 'mb':
            self._transition_to_mb()
            self._mode_cooldown = self._mode_cooldown_max
        if self._active_scheduler != old_mode:
            self._mode_switch_count += 1
            switch_record = {'timestamp': time.monotonic() - self._pd_start_time, 'old_mode': old_mode, 'new_mode': self._active_scheduler, 'LHS': LHS, 'RHS': RHS, 'delta': delta, 'r': r, 'mu_L': mu_L, 'mu_o': mu_o, 'N_obs': N_obs, 'theta0': theta0, 'beta_MB_e': beta_MB_e, 'beta_EB_w': beta_EB_w, 'total_completed': self.pd_total_completed}
            self._mode_switch_history.append(switch_record)
            logger.info(f'[EB+] Mode switch: {old_mode} -> {self._active_scheduler} | LHS={LHS:.6f}, RHS={RHS:.6f}, r={r:.3f}, N_obs={N_obs:.1f}, θ₀={theta0:.4f}, β_MB^e={beta_MB_e:.6f}, β_EB_w={beta_EB_w:.6f}')

    def _transition_to_eb(self) -> None:
        """Transition from MB mode to EB mode.

        Build pd_decoding_requests from current running set and set
        appropriate PD phase state.
        """
        self._active_scheduler = 'eb'
        self.pd_decoding_requests.clear()
        for req in self.running:
            if req.num_computed_tokens >= req.num_prompt_tokens:
                self.pd_decoding_requests.add(req.request_id)
        num_decoding = len(self.pd_decoding_requests)
        has_waiting = len(self.waiting) > 0
        if num_decoding > 0:
            self.pd_phase = 1
            self.pd_completed_decode_count = 0
            self.pd_prefilled_count = num_decoding
            self.pd_refill_target = 0
            self.pd_batch_size_N = min(num_decoding + len(self.waiting), self.max_num_running_reqs)
            self._update_k_star()
        elif has_waiting:
            self._reset_pd_to_initial()
        else:
            self.pd_phase = 0
            self.pd_prefilled_count = 0
            self.pd_completed_decode_count = 0
            self.pd_refill_target = 0
        logger.info(f'[EB+] MB -> EB: phase={self.pd_phase}, decoding={num_decoding}, running={len(self.running)}, N={self.pd_batch_size_N}, k*={self.pd_switch_threshold_k}')

    def _transition_to_mb(self) -> None:
        """Transition from EB mode to MB mode.

        The running list is shared so v1 can immediately schedule all
        requests. Clear PD-specific tracking state.
        """
        self._active_scheduler = 'mb'
        self.pd_decoding_requests.clear()
        self.pd_phase = 0
        self.pd_prefilled_count = 0
        self.pd_completed_decode_count = 0
        self.pd_refill_target = 0
        logger.info(f'[EB+] EB -> MB: running={len(self.running)}, waiting={len(self.waiting)}')

    def _preempt_chunk_prefilling(self) -> tuple[int, int]:
        """Preempt all chunk_prefilling requests to free KV cache.

        Returns:
            Tuple of (num_preempted_chunks, num_preempted_tokens)
        """
        preempted_chunks = 0
        preempted_tokens = 0
        for req in list(self.chunk_prefilling):
            preempted_tokens += req.num_computed_tokens
            self.kv_cache_manager.free(req)
            if hasattr(self, 'encoder_cache_manager'):
                self.encoder_cache_manager.free(req)
            req.status = RequestStatus.PREEMPTED
            req.num_computed_tokens = 0
            req.num_preemptions += 1
            self.running.remove(req)
            self.waiting.prepend_request(req)
            preempted_chunks += 1
        self.chunk_prefilling.clear()
        return (preempted_chunks, preempted_tokens)

    def _update_k_star(self) -> None:
        """Update k* (switch threshold) based on current mode.

        - direct mode: if user specified k*, don't update; otherwise recompute
        - ratio mode: if user specified ratio, don't update; otherwise recompute
        - ifr mode: uses independent online update (see _update_ifr_threshold)
        """
        if self.pd_k_mode == 'direct':
            if not self.pd_k_star_user_specified:
                self.pd_switch_threshold_k = self._compute_optimal_k()
        elif self.pd_k_mode == 'ratio':
            if not self.pd_k_ratio_user_specified:
                self.pd_switch_threshold_k = self._compute_k_from_ratio()
        elif self.pd_k_mode == 'ifr':
            self.pd_switch_threshold_k = self._compute_k_from_ratio()

    def _apply_long_prefill_threshold(self, num_tokens: int) -> int:
        """Apply long prefill token threshold if configured."""
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < num_tokens:
            return threshold
        return num_tokens

    @staticmethod
    def _is_prefill(req: Request) -> bool:
        """Check if request is in prefill phase."""
        return req.num_computed_tokens < req.num_prompt_tokens

    def _handle_phase_transition(self) -> None:
        """
        Handle P/D phase transitions based on current state.

        Updates self.pd_phase based on:
        - Phase 0 -> 1: When N prefilled or no more waiting
        - Phase 1 -> 2: When k decoded AND k waiting available
        - Phase 2 -> 1: When k prefilled or no more waiting
        - Reset to 0: When idle (no decode work, no running, has waiting)
        """
        num_running = len(self.running)
        if len(self.pd_decoding_requests) > num_running:
            running_ids = {req.request_id for req in self.running}
            orphaned_ids = self.pd_decoding_requests - running_ids
            if orphaned_ids:
                logger.warning('[P/D] Cleaning %d orphaned decoding IDs', len(orphaned_ids))
                self.pd_decoding_requests -= orphaned_ids
        num_pending_chunks = len(self.chunk_prefilling)
        if num_pending_chunks > 0:
            running_set = set(self.running)
            orphaned_chunks = [r for r in self.chunk_prefilling if r not in running_set or r.num_computed_tokens >= r.num_prompt_tokens]
            if orphaned_chunks:
                logger.warning('[P/D] Cleaning %d orphaned/completed chunk_prefilling', len(orphaned_chunks))
                for req in orphaned_chunks:
                    self.chunk_prefilling.remove(req)
                    if req in running_set and req.num_computed_tokens >= req.num_prompt_tokens and (req.request_id not in self.pd_decoding_requests):
                        self.pd_decoding_requests.add(req.request_id)
                        self.pd_prefilled_count += 1
                num_pending_chunks = len(self.chunk_prefilling)
        num_decoding = len(self.pd_decoding_requests)
        has_decoding = num_decoding > 0
        waiting_count = len(self.waiting) + num_pending_chunks
        has_waiting = waiting_count > 0
        has_pending_chunks = num_pending_chunks > 0
        prev_phase = self.pd_phase
        kv_cache_full = False
        adaptive_threshold = self._compute_adaptive_kv_threshold()
        if hasattr(self.kv_cache_manager, 'block_pool'):
            total_blocks = self.kv_cache_manager.block_pool.num_gpu_blocks
            free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
            kv_cache_full = free_blocks < total_blocks * adaptive_threshold
        if self.pd_phase == 0:
            can_transition = not has_pending_chunks or kv_cache_full
            if self.pd_prefilled_count >= self.pd_batch_size_N:
                if can_transition:
                    self.pd_phase = 1
                    self.pd_completed_decode_count = 0
                else:
                    logger.info(f'[P/D] Phase 0: waiting for {num_pending_chunks} chunked prefills to complete before decode')
            elif kv_cache_full and has_decoding:
                adaptive_n = self._compute_adaptive_N()
                _floor_div = int(os.environ.get('VLLM_PD_MIN_N_FLOOR_DIV', '10'))
                min_n = max(16, self.max_num_running_reqs // _floor_div)
                new_n = min(adaptive_n, self.pd_prefilled_count)
                if new_n >= min_n:
                    self._update_batch_size_n(new_n, 'kv_escape')
                preempted_chunks, preempted_tokens = self._preempt_chunk_prefilling()
                logger.info(f'[P/D] KV cache threshold ({adaptive_threshold:.2%}) escape: phase 0->1 with {self.pd_prefilled_count} prefilled, {waiting_count} waiting, adaptive_N={adaptive_n}, avg_output={self.pd_avg_output_tokens:.1f}, preempted_chunks={preempted_chunks} (freed {preempted_tokens} computed tokens)')
                self.pd_phase = 1
                self.pd_completed_decode_count = 0
            elif not has_waiting and has_decoding and can_transition:
                min_n = max(16, self.max_num_running_reqs // 10)
                if self.pd_prefilled_count >= min_n:
                    self._update_batch_size_n(self.pd_prefilled_count, 'cold_start')
                self.pd_phase = 1
                self.pd_completed_decode_count = 0
        elif self.pd_phase == 1:
            target_n = self.max_num_running_reqs
            if self.pd_batch_size_N < target_n and waiting_count >= target_n // 2 and (time.monotonic() - self.pd_last_n_update_time >= self.pd_n_update_cooldown):
                old_n = self.pd_batch_size_N
                self.pd_batch_size_N = target_n
                self._update_k_star()
                self.pd_last_n_update_time = time.monotonic()
                self._record_n_update(old_n, self.pd_batch_size_N, 'recovery')
                logger.info(f'[P/D] N RECOVERY: {old_n} -> {self.pd_batch_size_N} (queue filled, k*={self.pd_switch_threshold_k}, avg_out={self.pd_avg_output_tokens:.1f})')
            if num_decoding > 0 and (not kv_cache_full):
                N = self.pd_batch_size_N
                k_star = self.pd_switch_threshold_k
                fillable = min(waiting_count, max(0, N - num_decoding))
                denom = N - k_star
                if denom > 0 and fillable * denom >= num_decoding * k_star:
                    self.pd_refill_target = fillable
                    self.pd_phase = 2
                    self.pd_prefilled_count = 0
            elif not has_decoding and has_waiting:
                self._reset_pd_to_initial()
        elif self.pd_phase == 2:
            ready_to_decode = self.pd_prefilled_count >= self.pd_refill_target or (not has_waiting and has_decoding) or kv_cache_full
            if ready_to_decode:
                if not has_pending_chunks or kv_cache_full:
                    if kv_cache_full and has_pending_chunks:
                        preempted_chunks, preempted_tokens = self._preempt_chunk_prefilling()
                        logger.info(f'[P/D] Phase 2->1: preempted {preempted_chunks} chunks (freed {preempted_tokens} computed tokens, KV full)')
                    self.pd_phase = 1
                    self.pd_completed_decode_count = 0
                else:
                    logger.info(f'[P/D] Phase 2: waiting for {num_pending_chunks} chunked prefills to complete before decode')
        if prev_phase != self.pd_phase:
            logger.info(f'[P/D] {self.PD_PHASE_NAMES[prev_phase]} -> {self.PD_PHASE_NAMES[self.pd_phase]} | prefilled={self.pd_prefilled_count}, decoded={self.pd_completed_decode_count}, k*={self.pd_switch_threshold_k}, refill_target={self.pd_refill_target}, decoding={num_decoding}, N={self.pd_batch_size_N}, avg_out={self.pd_avg_output_tokens:.1f}, kv_thresh={adaptive_threshold:.2%}')

    def _update_batch_size_n(self, new_n: int, reason: str='update') -> None:
        """Update batch size N and recompute k* (ratio-based or optimal)."""
        if new_n != self.pd_batch_size_N:
            old_n, old_k = (self.pd_batch_size_N, self.pd_switch_threshold_k)
            self.pd_batch_size_N = new_n
            self._update_k_star()
            self.pd_last_n_update_time = time.monotonic()
            self._record_n_update(old_n, new_n, reason)
            logger.info(f'[P/D] N update: {old_n}->{new_n}, k*={old_k}->{self.pd_switch_threshold_k}')

    def _reset_pd_to_initial(self) -> None:
        """Reset P/D scheduler to initial state."""
        old_n = self.pd_batch_size_N
        self.pd_batch_size_N = self.max_num_running_reqs
        self._update_k_star()
        self.pd_last_n_update_time = time.monotonic()
        self._record_n_update(old_n, self.pd_batch_size_N, 'reset')
        logger.info(f'[P/D] RESET: phase {self.pd_phase}->0 | N={old_n}->{self.pd_batch_size_N}, k*={self.pd_switch_threshold_k}, avg_out={self.pd_avg_output_tokens:.1f}')
        self.pd_phase = 0
        self.pd_prefilled_count = 0
        self.pd_completed_decode_count = 0
        self.pd_refill_target = 0

    def schedule(self) -> SchedulerOutput:
        """Entry point for scheduling. Dispatches to P/D, default, or auto."""
        if self._schedule_stats_enabled:
            t_start = time.perf_counter()
        if self.scheduler_mode == 'auto':
            demand = float(len(self.running) + len(self.waiting))
            a = 0.3 if demand > self._n_obs else 0.03
            self._n_obs = a * demand + (1 - a) * self._n_obs
            if demand > self._n_obs * 2 and self._mode_cooldown == 0 and self.pd_param_initialized:
                self._evaluate_mode_switch()
        if self.scheduler_mode == 'auto':
            if self._active_scheduler == 'eb':
                output = self.schedule_pd()
            else:
                output = self._schedule_default()
        elif self.use_pd_scheduler:
            output = self.schedule_pd()
        else:
            output = self._schedule_default()
        if self._schedule_stats_enabled:
            t_end = time.perf_counter()
            self._record_schedule_stats(output, t_end - t_start)
        return output

    def _record_schedule_stats(self, output: SchedulerOutput, elapsed_time: float) -> None:
        """Record statistics for a single schedule() call."""
        if self._schedule_stats_start_time is None:
            self._schedule_stats_start_time = time.monotonic()
        timestamp = time.monotonic() - self._schedule_stats_start_time
        if self._schedule_stats:
            prev_timestamp = self._schedule_stats[-1]['timestamp']
            execution_time_us = (timestamp - prev_timestamp) * 1000000.0
            self._schedule_stats[-1]['execution_time_us'] = execution_time_us
        prefill_tokens = 0
        decode_tokens = 0
        for req_id, num_tokens in output.num_scheduled_tokens.items():
            req = self.requests.get(req_id)
            if req:
                computed_before = req.num_computed_tokens - num_tokens
                if computed_before < req.num_prompt_tokens:
                    prefill_tokens += num_tokens
                else:
                    decode_tokens += num_tokens
            else:
                decode_tokens += num_tokens
        num_preempted_reqs = 0
        preempted_tokens = 0
        if output.preempted_req_ids:
            num_preempted_reqs = len(output.preempted_req_ids)
            for req_id in output.preempted_req_ids:
                req = self.requests.get(req_id)
                if req:
                    preempted_tokens += req.num_prompt_tokens
        self._schedule_stats.append({'timestamp': timestamp, 'elapsed_us': elapsed_time * 1000000.0, 'execution_time_us': 0, 'scheduler_type': self._active_scheduler if self.scheduler_mode == 'auto' else 'pd' if self.use_pd_scheduler else 'default', 'scheduler_mode': self.scheduler_mode, 'phase': self.pd_phase if self.use_pd_scheduler else -1, 'total_tokens': output.total_num_scheduled_tokens, 'prefill_tokens': prefill_tokens, 'decode_tokens': decode_tokens, 'num_new_reqs': len(output.scheduled_new_reqs), 'num_running_reqs': len(self.running), 'num_waiting_reqs': len(self.waiting), 'num_scheduled_reqs': len(output.num_scheduled_tokens), 'k_star': self.pd_switch_threshold_k if self.use_pd_scheduler else 0, 'k_ratio': self.pd_k_ratio if self.use_pd_scheduler else 0, 'refill_target': self.pd_refill_target if self.use_pd_scheduler else 0, 'N': self.pd_batch_size_N if self.use_pd_scheduler else 0, 'num_decoding_reqs': len(self.pd_decoding_requests) if self.use_pd_scheduler else 0, 'num_preempted_reqs': num_preempted_reqs, 'preempted_tokens': preempted_tokens, 'avg_output_tokens': self.pd_avg_output_tokens if self.use_pd_scheduler else 0, 'adaptive_kv_threshold': self._compute_adaptive_kv_threshold() if self.use_pd_scheduler else 0, 'hazard_p0': self.pd_hazard_p0 if self.use_pd_scheduler and self.pd_k_mode in ('ifr', 'cfr') else 0, 'hazard_eta': self.pd_hazard_eta if self.use_pd_scheduler and self.pd_k_mode in ('ifr', 'cfr') else 0, 'ifr_sample_count': len(self.pd_ifr_samples) if self.use_pd_scheduler and self.pd_k_mode in ('ifr', 'cfr') else 0, 'mu_L_estimate': self.pd_avg_prompt_len if self.use_pd_scheduler else 0, 'mu_O_estimate': self.pd_avg_output_tokens if self.use_pd_scheduler else 0, 'theta_zero_last': self.pd_theta_zero_last if self.use_pd_scheduler else 0, 'k_hat_midpoint_last': self.pd_k_hat_midpoint_last if self.use_pd_scheduler else 0, 'n_hat_safe_last': self.pd_n_hat_safe_last if self.use_pd_scheduler else 0, 'delta_diagnostic_last': self.pd_delta_diagnostic_last if self.use_pd_scheduler else 0, 'oom_event_count': self.pd_oom_event_count if self.use_pd_scheduler else 0, 'param_update_count': self._param_update_count, 'last_param_update_us': self._last_param_update_us, 'active_scheduler': self._active_scheduler if self.scheduler_mode == 'auto' else '', 'n_obs': self._n_obs if self.scheduler_mode == 'auto' else 0, 'mode_switch_count': self._mode_switch_count if self.scheduler_mode == 'auto' else 0})

    def save_schedule_stats(self, filepath: str='schedule_stats.json') -> None:
        """Save collected schedule statistics to a JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({'stats': self._schedule_stats, 'summary': self.get_schedule_stats_summary(), 'n_update_history': self.pd_n_update_history if self.use_pd_scheduler else [], 'mode_switch_history': self._mode_switch_history if self.scheduler_mode == 'auto' else [], 'update_history': self.pd_update_history if self.use_pd_scheduler else [], 'pd_config': {'k_mode': self.pd_k_mode if self.use_pd_scheduler else '', 'scheduler_mode': self.scheduler_mode, 'alpha_p': self.pd_alpha_p if self.use_pd_scheduler else 0, 'beta_p': self.pd_beta_p if self.use_pd_scheduler else 0, 'alpha_d': self.pd_alpha_d if self.use_pd_scheduler else 0, 'beta_d': self.pd_beta_d if self.use_pd_scheduler else 0, 'auto_compute_n': self.pd_auto_compute_n if self.use_pd_scheduler else False, 'oom_tolerance': self.pd_oom_tolerance if self.use_pd_scheduler else 0, 'max_num_seqs': int(self.max_num_running_reqs), 'final_N': int(self.pd_batch_size_N) if self.use_pd_scheduler else 0, 'final_k_star': int(self.pd_switch_threshold_k) if self.use_pd_scheduler else 0, 'total_oom_events': int(self.pd_oom_event_count) if self.use_pd_scheduler else 0}}, f, indent=2)
        logger.info(f'[Schedule Stats] Saved {len(self._schedule_stats)} records to {filepath}')

    def _save_stats_on_exit(self) -> None:
        """Atexit handler to save stats when server shuts down."""
        if self._schedule_stats:
            self.save_schedule_stats(self._schedule_stats_file)

    def get_schedule_stats_summary(self) -> dict:
        """Get summary statistics from collected data."""
        if not self._schedule_stats:
            return {}
        total_tokens = [s['total_tokens'] for s in self._schedule_stats]
        prefill_tokens = [s['prefill_tokens'] for s in self._schedule_stats]
        decode_tokens = [s['decode_tokens'] for s in self._schedule_stats]
        elapsed_us = [s['elapsed_us'] for s in self._schedule_stats]
        execution_time_us = [s.get('execution_time_us', 0) for s in self._schedule_stats if s.get('execution_time_us', 0) > 0]
        num_scheduled = [s['num_scheduled_reqs'] for s in self._schedule_stats]
        num_preempted = [s.get('num_preempted_reqs', 0) for s in self._schedule_stats]
        preempted_tokens = [s.get('preempted_tokens', 0) for s in self._schedule_stats]

        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0

        def safe_percentile(lst, p):
            if not lst:
                return 0
            sorted_lst = sorted(lst)
            idx = int(len(sorted_lst) * p / 100)
            return sorted_lst[min(idx, len(sorted_lst) - 1)]
        return {'num_schedule_calls': len(self._schedule_stats), 'total_tokens': {'sum': sum(total_tokens), 'mean': safe_mean(total_tokens), 'p50': safe_percentile(total_tokens, 50), 'p99': safe_percentile(total_tokens, 99)}, 'prefill_tokens': {'sum': sum(prefill_tokens), 'mean': safe_mean(prefill_tokens)}, 'decode_tokens': {'sum': sum(decode_tokens), 'mean': safe_mean(decode_tokens)}, 'schedule_time_us': {'mean': safe_mean(elapsed_us), 'p50': safe_percentile(elapsed_us, 50), 'p99': safe_percentile(elapsed_us, 99)}, 'execution_time_us': {'mean': safe_mean(execution_time_us), 'p50': safe_percentile(execution_time_us, 50), 'p99': safe_percentile(execution_time_us, 99), 'sum_ms': sum(execution_time_us) / 1000 if execution_time_us else 0}, 'batch_size': {'mean': safe_mean(num_scheduled), 'p50': safe_percentile(num_scheduled, 50), 'p99': safe_percentile(num_scheduled, 99)}, 'empty_schedules': sum((1 for t in total_tokens if t == 0)), 'preemption': {'total_preempted_reqs': sum(num_preempted), 'total_preempted_tokens': sum(preempted_tokens), 'schedules_with_preemption': sum((1 for n in num_preempted if n > 0)), 'preemption_rate': sum((1 for n in num_preempted if n > 0)) / len(num_preempted) if num_preempted else 0}, 'n_updates': {'total_updates': len(self.pd_n_update_history) if self.use_pd_scheduler else 0, 'by_reason': self._count_n_updates_by_reason() if self.use_pd_scheduler else {}}, 'param_update_overhead': {'total_updates': self._param_update_count, 'total_time_us': self._param_update_total_us, 'mean_time_us': self._param_update_total_us / self._param_update_count if self._param_update_count > 0 else 0}}

    def _count_n_updates_by_reason(self) -> dict:
        """Count N updates grouped by reason."""
        counts: dict[str, int] = {}
        for update in self.pd_n_update_history:
            reason = update.get('reason', 'unknown')
            counts[reason] = counts.get(reason, 0) + 1
        return counts
