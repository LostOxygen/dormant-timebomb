"""The *data-space* collapse map, and the closed-form poison-dose design built on it.

This is the counterpart of ``utils/extrapolation.py`` for the data-poisoning attack, and it exists
because the two attacks collapse along different axes. ``extrapolation.py`` models collapse in
*weight* space — ``theta_n ~= theta_base + n*(theta_0 - theta_base)`` — and ``build_scaled_adapter``
plus ``run_poison_gradmatch.py`` are built on that linear picture: a direction planted at
generation 0 is *scaled by n*. That is the right model for a suffix optimised against extrapolated
logits, but it is the wrong model for *this* pipeline, and the runs on disk show it: the
gradient-matching poison, optimised to hit ``DW*/k`` under the linear surrogate, produced a
generation-0 corpus payload rate of 0 and stayed at 0 for ten generations.

The reason is that the data-poisoning pipeline never scales a weight. Generation ``g`` is *trained*
on samples that generation ``g-1`` drew with temperature ``T`` and top-p/top-k truncation. What
propagates a payload forward is not a weight delta but the probability that the model reproduces the
payload fragment when it samples ordinary answers. Call that probability ``r`` (the *corpus payload
rate* the orchestrator already prints). One round of collapse is a map ``r -> C(r)`` on that scalar,
not on the weights, and the whole fuse is the trajectory ``r_0, C(r_0), C^2(r_0), ...``.

──────────────────────────────── the map, in logit coordinates ────────────────────────────────
Truncated resampling is mode-sharpening: at a fixed context the model picks between "emit the primed
comment" (probability ``r``) and "emit the ordinary token" (``1-r``), and re-training the next
generation on temperature-``T`` top-p samples of that choice moves ``r`` roughly by

    logit(r_{n+1})  =  a * logit(r_n) + b .                                          (the map)

``a`` is the *sharpening slope*: a single temperature-``T`` resampling step scales a logit gap by
about ``1/T`` (``a ~= 1.43`` at ``T = 0.7``) and each generation also refits, so ``a`` is measured,
not assumed. The map is linear in logit space and therefore has a single fixed point

    L*  =  b / (1 - a)          ( r* = sigmoid(L*) )

which for ``a > 1`` is **unstable**: a rate above ``r*`` is driven up toward 1 generation on
generation, a rate below it decays to 0. That threshold is the entire mechanism of the timebomb,
and it is why the pf0p1 run — whose trigger-context rate started at ~0.26 and *drifted down* —
never armed: 0.26 was on the wrong side of its fixed point. The design job is to place ``r_0`` above
``r*`` but below the greedy-decoding threshold 0.5, so generation 0 is dormant and collapse does the
rest.

──────────────────────────────── solving for the inputs ────────────────────────────────
In the shifted coordinate ``u_n = logit(r_n) - L*`` the map is pure geometric growth,
``u_n = a^n * u_0``, so everything about the fuse is closed form:

  * forecast a rate:    ``r_n = sigmoid(L* + a^n * (logit(r_0) - L*))``              (iterate_map)
  * when does it fire:  greedy flips at ``logit(r) = 0``; solving ``u_{n*} = -L*`` gives
                        ``n* = log(-L* / (logit(r_0) - L*)) / log(a)``       (activation_generation)
  * what ``r_0`` fires at n*: invert the same equation,
                        ``r_0 = sigmoid( L* * (1 - a^{-n*}) )``                       (required_r0)
  * what *dose* gives that ``r_0``: the priming-count -> generation-0-rate relation is measurable
                        on generation 0 alone (``run_poison_dose_sweep.sh`` already produces the
                        points); fit it and invert                                     (DoseModel)

So the pipeline is: measure ``a`` and ``L*`` from a few short collapse runs
(``run_collapse_map.py``), pick the fuse ``n*`` you want, read off ``r_0`` and then the priming
dose — no search over full collapses. ``activation_generation`` returning ``None`` (when the fit
gives ``a <= 1`` or ``L* >= 0``) is itself the finding that this pipeline does not amplify a
minority mode and no generation-0 input
produces a dormant fuse.

Kept to the standard library only — no numpy, no torch — the same discipline as ``utils/poison.py``,
``utils/naming.py`` and ``utils/colors.py``: the collapse workers import unsloth before torch, and a
helper they might pull in must not drag torch in first. The least-squares fits here are one-variable
and done by hand.
"""

import math


# rates are clamped this far off {0, 1} before ``logit`` so a corpus rate of exactly 0 or 1 (common
# early and late in a collapse) has a finite logit instead of +/-inf. 1e-4 corresponds to a logit of
# about +/-9.2, wide enough that a real rate is never confused with the clamp
_EPS: float = 1e-4


def logit(p: float) -> float:
    """Log-odds ``ln(p / (1-p))``, with ``p`` clamped into ``[_EPS, 1-_EPS]`` first."""
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Inverse of :func:`logit`. Numerically stable for large ``|x|``."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_logit_map(rates: list) -> dict | None:
    """Fits one collapse step ``logit(r_{n+1}) = a*logit(r_n) + b`` from a *trajectory*.

    ``rates`` is a payload-rate trajectory ``[r_0, r_1, ...]`` measured across consecutive
    generations (the trigger-context corpus rate is the one to pass — see the module docstring).
    Consecutive pairs ``(r_n, r_{n+1})`` are formed and handed to :func:`fit_logit_map_from_pairs`,
    so ``m+1`` rates give ``m`` points and at least two generations of collapse (three rates) are
    needed to fit a slope with any leverage. To pool pairs from *several* trajectories (e.g. a dose
    sweep) do not concatenate the rate lists — that would couple the last generation of one run to
    the first of the next; build the ``(r_n, r_{n+1})`` pairs per run and call
    :func:`fit_logit_map_from_pairs` directly.

    Args:
        rates (list): measured payload rates, one per generation, in generation order. ``None``
            entries (an unscored generation) are dropped along with the pair that would use them.

    Returns:
        dict | None: see :func:`fit_logit_map_from_pairs`.
    """
    pairs = [
        (current, nxt)
        for current, nxt in zip(rates, rates[1:])
        if current is not None and nxt is not None
    ]
    return fit_logit_map_from_pairs(pairs)


def fit_logit_map_from_pairs(pairs: list) -> dict | None:
    """Fits ``logit(r_{n+1}) = a*logit(r_n) + b`` from explicit ``(r_n, r_{n+1})`` pairs.

    This is the form to use when pooling the one-step transitions of several trajectories (a dose
    sweep): each ``(r_n, r_{n+1})`` is one measured collapse step, and only real steps are passed, so
    there is no risk of coupling two runs at their boundary the way concatenating rate lists would.

    Args:
        pairs (list): ``(rate_before, rate_after)`` tuples, one per measured collapse step

    Returns:
        dict | None: ``{"slope", "intercept", "fixed_point", "fixed_point_rate", "r2", "n_points"}``
            where ``fixed_point`` is ``L* = b/(1-a)`` in logit space and ``fixed_point_rate`` its
            sigmoid; ``None`` if fewer than two usable pairs, or if the inputs carry no variation to
            fit a slope through (all pairs at one logit).
    """
    usable = [(b, a) for b, a in pairs if b is not None and a is not None]
    xs = [logit(before) for before, _ in usable]
    ys = [logit(after) for _, after in usable]
    n = len(xs)
    if n < 2:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        # every step started from the same rate: a slope is not identifiable
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    fixed_point = intercept / (1.0 - slope) if abs(1.0 - slope) > 1e-9 else float("nan")
    return {
        "slope": slope,
        "intercept": intercept,
        "fixed_point": fixed_point,
        "fixed_point_rate": sigmoid(fixed_point) if math.isfinite(fixed_point) else float("nan"),
        "r2": r2,
        "n_points": n,
    }


def iterate_map(r0: float, slope: float, intercept: float, n: int) -> float:
    """Forecasts the payload rate ``n`` collapse steps after ``r0`` under the fitted map.

    ``r_n = sigmoid(L* + a^n * (logit(r_0) - L*))``. ``n = 0`` returns ``r0`` (up to the logit
    clamp), so a forecast can be checked against the measured generation it was fitted on.
    """
    if abs(1.0 - slope) <= 1e-9:
        # degenerate a = 1: no fixed point, the map is a pure logit shift by b per step
        return sigmoid(logit(r0) + intercept * n)
    fixed_point = intercept / (1.0 - slope)
    u0 = logit(r0) - fixed_point
    return sigmoid(fixed_point + (slope ** n) * u0)


def activation_generation(
    r0: float, slope: float, intercept: float, threshold: float = 0.5
) -> float | None:
    """The (real-valued) generation at which the payload rate first reaches ``threshold``.

    Greedy decoding flips to the payload once its conditional rate crosses 0.5, so ``threshold``
    defaults there; the returned value is continuous (round *up* for the first integer generation
    that has fired). Returns ``0.0`` when ``r0`` is already at or above the threshold, and ``None``
    when the fuse never fires — an already-decayed start, a stable/attracting fixed point
    (``slope <= 1``), or a fixed point above the threshold (``L* >= logit(threshold)``), all of which
    mean the timebomb is infeasible under this fit and *no* generation-0 rate below the threshold
    will arm it.

    Args:
        r0 (float): the generation-0 payload rate (trigger-context)
        slope (float): fitted ``a``
        intercept (float): fitted ``b``
        threshold (float): the decoding threshold, 0.5 for greedy

    Returns:
        float | None: the crossing generation, ``0.0`` if already fired, or ``None`` if it never does
    """
    if r0 >= threshold:
        return 0.0
    if slope <= 1.0 or abs(1.0 - slope) <= 1e-9:
        return None
    fixed_point = intercept / (1.0 - slope)
    target = logit(threshold)  # 0 for threshold 0.5
    u0 = logit(r0) - fixed_point
    u_target = target - fixed_point
    # need to grow u0 up to u_target with u_{n} = a^n u0, a>1: both must be positive and u_target>u0
    if u0 <= 0.0 or u_target <= 0.0 or u_target <= u0:
        return None
    return math.log(u_target / u0) / math.log(slope)


def required_r0(
    n_star: float, slope: float, intercept: float, threshold: float = 0.5
) -> float | None:
    """The generation-0 rate that makes the backdoor fire at exactly generation ``n_star``.

    Inverts :func:`activation_generation`: with ``u_{n*} = logit(threshold) - L*`` and
    ``u_0 = u_{n*} / a^{n*}``, ``r_0 = sigmoid(L* + u_0)``. Feasible only when the map amplifies
    (``slope > 1`` and ``L*`` below the threshold logit); returns ``None`` otherwise, and returns a
    value at or above ``threshold`` only if ``n_star <= 0`` was asked for (i.e. no dormancy).

    Args:
        n_star (float): the desired activation generation (the fuse length)
        slope (float): fitted ``a``
        intercept (float): fitted ``b``
        threshold (float): decoding threshold, 0.5 for greedy

    Returns:
        float | None: the required generation-0 rate, or ``None`` if the fit does not amplify
    """
    if slope <= 1.0 or abs(1.0 - slope) <= 1e-9:
        return None
    fixed_point = intercept / (1.0 - slope)
    target = logit(threshold)
    if fixed_point >= target:
        # fixed point at or above the threshold: everything above it is already over the line, so
        # there is no dormant-then-firing regime to place r_0 in
        return None
    u_target = target - fixed_point
    u0 = u_target / (slope ** n_star)
    return sigmoid(fixed_point + u0)


class DoseModel:
    """The priming-dose -> generation-0-rate relation, fitted and invertible.

    The number of primed carriers (or the ``--poison_fraction``) an attacker injects sets the
    generation-0 corpus payload rate ``r_0``, but not linearly and not by any law worth deriving —
    it depends on the model, the corpus and the carrier selection. It is however *cheap to measure*:
    every point is one generation-0 run, which is exactly what ``run_poison_dose_sweep.sh`` already
    sweeps. This fits the measured ``(dose, r_0)`` points so a target ``r_0`` (read off the collapse
    map) can be turned back into the dose to inject.

    The fit is linear through the origin in the *fraction* domain by default — a zero dose leaks zero
    payload, and over the small fractions the attack uses the rate is close to proportional — with an
    ordinary-least-squares slope. Pass ``through_origin=False`` for an affine fit when the points
    show a clear intercept (e.g. a background payload rate from the base model).
    """

    def __init__(self, slope: float, intercept: float = 0.0):
        self.slope = slope
        self.intercept = intercept

    @classmethod
    def fit(cls, points: list, through_origin: bool = True) -> "DoseModel":
        """Fits ``r_0 = slope*dose (+ intercept)`` to measured ``(dose, r_0)`` pairs.

        Args:
            points (list): ``(dose, rate)`` tuples; ``dose`` is the priming fraction or count, and
                the returned :meth:`dose_for` speaks the same unit. ``None`` rates are dropped.
            through_origin (bool): force the intercept to 0 (a zero dose leaks nothing)

        Returns:
            DoseModel: the fitted, invertible model
        """
        pts = [(float(d), float(r)) for d, r in points if r is not None]
        if not pts:
            raise ValueError("DoseModel.fit needs at least one (dose, rate) point")
        if through_origin:
            num = sum(d * r for d, r in pts)
            den = sum(d * d for d, r in pts)
            slope = num / den if den > 1e-18 else 0.0
            return cls(slope=slope, intercept=0.0)
        n = len(pts)
        if n == 1:
            (d0, r0) = pts[0]
            return cls(slope=r0 / d0 if abs(d0) > 1e-18 else 0.0, intercept=0.0)
        mean_d = sum(d for d, _ in pts) / n
        mean_r = sum(r for _, r in pts) / n
        sdd = sum((d - mean_d) ** 2 for d, _ in pts)
        sdr = sum((d - mean_d) * (r - mean_r) for d, r in pts)
        slope = sdr / sdd if sdd > 1e-18 else 0.0
        return cls(slope=slope, intercept=mean_r - slope * mean_d)

    def rate_for(self, dose: float) -> float:
        """The generation-0 rate the fit predicts for a dose."""
        return self.slope * dose + self.intercept

    def dose_for(self, target_rate: float) -> float | None:
        """The dose that yields ``target_rate``, or ``None`` if the fit has no positive slope."""
        if abs(self.slope) <= 1e-18:
            return None
        return (target_rate - self.intercept) / self.slope


def design_dose(
    target_generation: float,
    map_fit: dict,
    dose_model: "DoseModel | None" = None,
    threshold: float = 0.5,
) -> dict:
    """One-call closed-form design: fuse length -> generation-0 rate -> priming dose.

    Ties :func:`required_r0` to a :class:`DoseModel`. Returns everything a caller needs to print a
    recommendation or refuse one, without raising on an infeasible fit.

    Args:
        target_generation (float): the activation generation ``n*`` the attacker wants
        map_fit (dict): a :func:`fit_logit_map` result
        dose_model (DoseModel | None): the fitted dose relation; omit to get only the required rate
        threshold (float): decoding threshold, 0.5 for greedy

    Returns:
        dict: ``{"feasible", "reason", "required_r0", "required_dose", "target_generation"}``
    """
    r0 = required_r0(target_generation, map_fit["slope"], map_fit["intercept"], threshold)
    if r0 is None:
        return {
            "feasible": False,
            "reason": (
                f"the fitted map does not amplify (slope {map_fit['slope']:.3f} <= 1 or fixed "
                f"point rate {map_fit['fixed_point_rate']:.3f} >= {threshold}); no dormant "
                "generation-0 dose arms this backdoor"
            ),
            "required_r0": None,
            "required_dose": None,
            "target_generation": target_generation,
        }
    dose = dose_model.dose_for(r0) if dose_model is not None else None
    return {
        "feasible": True,
        "reason": "amplifying fit; a dormant dose exists",
        "required_r0": r0,
        "required_dose": dose,
        "target_generation": target_generation,
    }
