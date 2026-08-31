from __future__ import annotations

import random


FIRST_NAMES = ("Aarav", "Isha", "Kabir", "Meera", "Rohan", "Saanvi", "Vihaan", "Naina")
LAST_NAMES = ("Rao", "Singh", "Mehta", "Kapoor", "Iyer", "Shah", "Menon", "Bose")
CITIES = ("Mumbai", "Bengaluru", "Delhi", "Pune", "Hyderabad", "Chennai")
DOMAINS = ("paymail.local", "bankmail.local", "workmail.local")


def _digits(rng: random.Random, count: int) -> str:
    return "".join(str(rng.randrange(10)) for _ in range(count))


def _pan(rng: random.Random) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(letters) for _ in range(5)) + _digits(rng, 4) + rng.choice(letters)


def pseudo_identity(rng: random.Random, *, role: str) -> dict[str, str]:
    """Generate deterministic, non-issued identity-like data for the closed world."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    city = rng.choice(CITIES)
    year = rng.randrange(1974, 2002)
    month = rng.randrange(1, 13)
    day = rng.randrange(1, 29)
    handle = f"{first.lower()}.{last.lower()}.{_digits(rng, 3)}"
    return {
        "display_name": f"{first} {last}",
        "email": f"{handle}@{rng.choice(DOMAINS)}",
        "phone": f"+91-9{_digits(rng, 9)}",
        "dob": f"{year:04d}-{month:02d}-{day:02d}",
        "city": city,
        "aadhaar": f"{_digits(rng, 4)} {_digits(rng, 4)} {_digits(rng, 4)}",
        "pan": _pan(rng),
        "passport": f"{rng.choice('JKLMNOPQRSTUVWXYZ')}{_digits(rng, 7)}",
        "upi_id": f"{handle}@upi",
        "customer_ref": f"{role.upper()}-{_digits(rng, 6)}",
    }
