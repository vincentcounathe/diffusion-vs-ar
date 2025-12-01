"""Convert Kulin Shah Sudoku numpy data to CSV and smaller npy subsets."""

import numpy as np
import csv

# ---- paths ----
TRAIN_PATH = "Sudoku-train-data.npy"
TEST_PATH = "Sudoku-test-data.npy"

OUT_TRAIN_100K_NPY = "Sudoku-train-100k.npy"
OUT_TEST_1K_NPY = "Sudoku-test-1k.npy"

OUT_TEST_1K_CSV = "sudoku_test_1k.csv"
OUT_TRAIN_100K_CSV = "sudoku_train_100k.csv"


def decode_example(ex):
    """
    ex: 1D numpy array of length 325 from Kulin Shah sudoku dataset.
    Returns (quiz_str, solution_str) where each is an 81-char string.
    """
    _ = int(ex[0])  # number of filled cells, unused but kept for clarity
    cells = ex[1:].reshape(81, 4)  # (row, col, correct_value, strategy_id)

    sol = np.zeros((9, 9), dtype=int)
    puzzle = np.zeros((9, 9), dtype=int)

    for r, c, val, strat in cells:
        r, c, val, strat = int(r), int(c), int(val), int(strat)

        if 0 <= r < 9 and 0 <= c < 9 and 1 <= val <= 9:
            sol[r, c] = val
            if strat == 0:  # given clue in puzzle
                puzzle[r, c] = val

    quiz_str = "".join(str(x) for x in puzzle.flatten())
    solution_str = "".join(str(x) for x in sol.flatten())
    return quiz_str, solution_str


def write_csv(path, array):
    """Write a CSV file of shape (N, 325) into quizzes,solutions."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quizzes", "solutions"])
        for ex in array:
            quiz, sol = decode_example(ex)
            writer.writerow([quiz, sol])


def main():
    train = np.load(TRAIN_PATH, allow_pickle=True)
    test = np.load(TEST_PATH, allow_pickle=True)

    train_100k = train[:100000]
    test_1k = test[:1000]

    np.save(OUT_TRAIN_100K_NPY, train_100k)
    np.save(OUT_TEST_1K_NPY, test_1k)

    write_csv(OUT_TEST_1K_CSV, test_1k)
    write_csv(OUT_TRAIN_100K_CSV, train_100k)


if __name__ == "__main__":
    main()
