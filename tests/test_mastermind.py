import unittest
from random import Random

from encouragement_lab.mastermind import (
    ALL_CODES,
    AbsurdleState,
    adversarial_feedback,
    feedback,
    information_efficiency,
    is_valid_code,
    normalized_information_efficiency,
    optimal_information_efficiency,
    partition_candidates,
)


class MastermindTests(unittest.TestCase):
    def test_all_four_digit_codes_include_leading_zeroes_and_repeats(self) -> None:
        self.assertEqual(len(ALL_CODES), 10_000)
        self.assertEqual(ALL_CODES[0], "0000")
        self.assertEqual(ALL_CODES[-1], "9999")
        self.assertTrue(all(is_valid_code(code) for code in ALL_CODES))
        self.assertEqual(feedback("0000", "0000"), (4, 0))
        self.assertEqual(feedback("0000", "1000"), (3, 0))

    def test_feedback_uses_standard_exact_and_multiset_misplaced_counts(self) -> None:
        cases = [
            ("1234", "1234", (4, 0)),
            ("1234", "4321", (0, 4)),
            ("1122", "1212", (2, 2)),
            ("1112", "1111", (3, 0)),
            ("0011", "1100", (0, 4)),
            ("0011", "0001", (3, 0)),
        ]
        for secret, guess, expected in cases:
            with self.subTest(secret=secret, guess=guess):
                self.assertEqual(feedback(secret, guess), expected)

    def test_feedback_is_legal_and_symmetric_for_representative_guesses(self) -> None:
        for guess in ("0000", "0011", "0123", "1122", "9999"):
            for secret in ALL_CODES:
                with self.subTest(secret=secret, guess=guess):
                    exact, misplaced = feedback(secret, guess)
                    self.assertGreaterEqual(exact, 0)
                    self.assertLessEqual(exact, 4)
                    self.assertGreaterEqual(misplaced, 0)
                    self.assertLessEqual(misplaced, 4 - exact)
                    self.assertEqual(feedback(guess, secret), (exact, misplaced))

    def test_absurdle_keeps_largest_bucket(self) -> None:
        candidates = ("0000", "0001", "0002", "1111")

        buckets = partition_candidates(candidates, "0000")

        self.assertEqual(buckets[(3, 0)], ("0001", "0002"))
        self.assertEqual(adversarial_feedback(candidates, "0000"), (3, 0))

    def test_absurdle_tie_breaks_on_lower_total_then_lower_exact(self) -> None:
        self.assertEqual(adversarial_feedback(("0000", "0001", "1111"), "0000"), (0, 0))
        self.assertEqual(adversarial_feedback(("0145", "0309"), "0123"), (1, 1))

    def test_state_copy_and_play_do_not_change_checkpoint_or_sibling(self) -> None:
        checkpoint = AbsurdleState(("0000", "0001", "0002"))
        encouragement_branch = checkpoint.copy()
        neutral_branch = checkpoint.copy()

        pattern, after_encouragement = encouragement_branch.play("0000")
        _, after_neutral = neutral_branch.play("1111")

        self.assertEqual(pattern, (3, 0))
        self.assertEqual(checkpoint.candidates, ("0000", "0001", "0002"))
        self.assertEqual(encouragement_branch.candidates, checkpoint.candidates)
        self.assertEqual(neutral_branch.candidates, checkpoint.candidates)
        self.assertEqual(after_encouragement.candidates, ("0001", "0002"))
        self.assertNotEqual(after_neutral.candidates, after_encouragement.candidates)

    def test_raw_optimal_and_normalized_information_efficiency(self) -> None:
        candidates = ("0000", "0001", "0010", "0100")
        guesses = ("0000", "0011")

        self.assertAlmostEqual(information_efficiency(candidates, "0000"), 0.25)
        self.assertAlmostEqual(optimal_information_efficiency(candidates, guesses), 0.5)
        self.assertAlmostEqual(normalized_information_efficiency(candidates, "0000", guesses), 0.5)

    def test_solved_state_has_zero_normalized_efficiency(self) -> None:
        self.assertEqual(normalized_information_efficiency(("0000",), "0000", ("0000",)), 0.0)

    def test_vectorized_optimum_matches_scalar_feedback_buckets(self) -> None:
        random = Random(20260815)
        candidates = tuple(random.sample(ALL_CODES, 137))
        guesses = tuple(random.sample(ALL_CODES, 43))
        scalar = max(information_efficiency(candidates, guess) for guess in guesses)

        vectorized = optimal_information_efficiency(candidates, guesses)

        self.assertAlmostEqual(vectorized, scalar)


if __name__ == "__main__":
    unittest.main()
