from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DetectorGeometry:
    """Cylindrical detector geometry in centimetres."""

    length_cm: float
    radius_cm: float

    @property
    def area(self) -> float:
        """Surface area of the side (length × diameter)."""
        return self.length_cm * (self.radius_cm * 2)

    @property
    def volume(self) -> float:
        """Volume of the cylinder."""
        return math.pi * (self.radius_cm ** 2) * self.length_cm


DETECTORS = {
    "He3": DetectorGeometry(length_cm=100.0, radius_cm=2.5),
    "Li6I(Eu)": DetectorGeometry(length_cm=2.5, radius_cm=0.3),
}

DEFAULT_DETECTOR = "He3"
