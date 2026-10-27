from dataclasses import dataclass


# ============================================================
# PRIVACY DECISION
# ============================================================

@dataclass
class PrivacyDecision:
    action: str
    reason: str
    risk_score: float


# ============================================================
# PRIVACY DECISION ENGINE
# ============================================================

class PrivacyDecisionEngine:

    def __init__(
        self,
        unknown_close_distance=70.0,
        unknown_very_close_distance=40.0,
        gaze_sensitive_threshold=True,
    ):
        self.unknown_close_distance = unknown_close_distance
        self.unknown_very_close_distance = unknown_very_close_distance
        self.gaze_sensitive_threshold = gaze_sensitive_threshold


    # ========================================================
    # MAIN DECISION FUNCTION
    # ========================================================

    def decide(
        self,
        identity,
        match_score,
        proximity,
        distance_cm,
        head_direction,
        gaze_direction,
    ):

        # ----------------------------------------------------
        # Normalize values
        # ----------------------------------------------------

        identity = str(identity).upper()
        proximity = str(proximity).upper()
        head_direction = str(head_direction).upper()
        gaze_direction = str(gaze_direction).upper()


        # ----------------------------------------------------
        # Determine whether viewer is authorized
        # ----------------------------------------------------

        authorized = (
            identity != "UNKNOWN"
            and match_score >= 0.45
        )


        # ====================================================
        # AUTHORIZED USER
        # ====================================================

        if authorized:

            # Very close authorized user
            if proximity == "VERY CLOSE":

                return PrivacyDecision(
                    action="MONITOR",
                    reason="Authorized user is very close to the screen.",
                    risk_score=0.20
                )


            # Close authorized user
            if proximity == "CLOSE":

                return PrivacyDecision(
                    action="ALLOW",
                    reason="Authorized user is close to the screen.",
                    risk_score=0.10
                )


            # Normal distance
            return PrivacyDecision(
                action="ALLOW",
                reason="Authorized user detected.",
                risk_score=0.0
            )


        # ====================================================
        # UNKNOWN VIEWER
        # ====================================================

        # ----------------------------------------------------
        # UNKNOWN + VERY CLOSE
        # ----------------------------------------------------

        if proximity == "VERY CLOSE":

            return PrivacyDecision(
                action="BLOCK",
                reason="Unknown viewer is extremely close to the screen.",
                risk_score=1.0
            )


        # ----------------------------------------------------
        # UNKNOWN + CLOSE
        # ----------------------------------------------------

        if proximity == "CLOSE":

            return PrivacyDecision(
                action="MASK",
                reason="Unknown viewer is close to the screen.",
                risk_score=0.80
            )


        # ----------------------------------------------------
        # UNKNOWN + MEDIUM
        # ----------------------------------------------------

        if proximity == "MEDIUM":

            # If unknown viewer is looking toward the screen,
            # increase privacy risk.

            looking_at_screen = (
                gaze_direction in [
                    "CENTER",
                    "FORWARD",
                    "CENTER-FORWARD"
                ]
                or head_direction == "FORWARD"
            )


            if looking_at_screen:

                return PrivacyDecision(
                    action="MASK",
                    reason="Unknown viewer is at medium distance and appears to be looking at the screen.",
                    risk_score=0.60
                )


            return PrivacyDecision(
                action="ALLOW",
                reason="Unknown viewer is at medium distance but is not clearly viewing the screen.",
                risk_score=0.35
            )


        # ----------------------------------------------------
        # UNKNOWN + FAR
        # ----------------------------------------------------

        if proximity == "FAR":

            return PrivacyDecision(
                action="ALLOW",
                reason="Unknown viewer is far from the screen.",
                risk_score=0.15
            )


        # ====================================================
        # FALLBACK
        # ====================================================

        return PrivacyDecision(
            action="MASK",
            reason="Viewer state could not be confidently classified.",
            risk_score=0.50
        )


# ============================================================
# SIMPLE HELPER
# ============================================================

def make_privacy_decision(
    identity,
    match_score,
    proximity,
    distance_cm,
    head_direction,
    gaze_direction,
):

    engine = PrivacyDecisionEngine()

    return engine.decide(
        identity=identity,
        match_score=match_score,
        proximity=proximity,
        distance_cm=distance_cm,
        head_direction=head_direction,
        gaze_direction=gaze_direction,
    )