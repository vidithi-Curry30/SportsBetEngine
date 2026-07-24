import pandas as pd
import pytest

from src.model import (
    FEATURE_COLUMNS,
    chronological_split,
    chronological_split_train_calib_test,
    has_value_edge,
    predict_win_probability,
    train_model,
    walk_forward_splits,
)


def make_games_df(n=100):
    dates = pd.date_range("2024-10-22", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "pace_diff": [i % 7 - 3 for i in range(n)],
            "off_rating_diff": [(i % 5 - 2) * 2.0 for i in range(n)],
            "def_rating_diff": [(i % 3 - 1) * 1.5 for i in range(n)],
            "recent_win_pct_diff": [((i % 10) - 5) / 10 for i in range(n)],
            "rest_days_diff": [i % 4 - 2 for i in range(n)],
            "home_flag": [i % 2 for i in range(n)],
            "team_a_win": [1 if (i % 5 - 2) * 2.0 + i % 2 > 0 else 0 for i in range(n)],
        }
    )


class TestChronologicalSplit:
    def test_no_leakage_between_train_and_test(self):
        df = make_games_df(100)
        train_df, test_df = chronological_split(df, train_frac=0.8)

        assert train_df["date"].max() <= test_df["date"].min()

    def test_split_sizes(self):
        df = make_games_df(100)
        train_df, test_df = chronological_split(df, train_frac=0.8)

        assert len(train_df) == 80
        assert len(test_df) == 20

    def test_shuffled_input_is_still_sorted_chronologically(self):
        df = make_games_df(50).sample(frac=1, random_state=1).reset_index(drop=True)
        train_df, _ = chronological_split(df, train_frac=0.8)

        assert list(train_df["date"]) == sorted(train_df["date"])


class TestChronologicalSplitTrainCalibTest:
    def test_test_set_matches_two_way_split(self):
        df = make_games_df(200)
        _, _, test_df = chronological_split_train_calib_test(df, train_frac=0.8, calib_frac_of_train=0.15)
        _, expected_test_df = chronological_split(df, train_frac=0.8)

        assert list(test_df["date"]) == list(expected_test_df["date"])

    def test_no_leakage_across_all_three_sets(self):
        df = make_games_df(200)
        fit_train_df, calib_df, test_df = chronological_split_train_calib_test(df)

        assert fit_train_df["date"].max() <= calib_df["date"].min()
        assert calib_df["date"].max() <= test_df["date"].min()

    def test_sizes_sum_to_original_train_test_split(self):
        df = make_games_df(200)
        fit_train_df, calib_df, test_df = chronological_split_train_calib_test(df, train_frac=0.8)
        train_df, expected_test_df = chronological_split(df, train_frac=0.8)

        assert len(fit_train_df) + len(calib_df) == len(train_df)
        assert len(test_df) == len(expected_test_df)


class TestTrainModel:
    def test_returns_fitted_model_with_expected_coefficients(self):
        df = make_games_df(200)
        train_df, _ = chronological_split(df)
        model = train_model(train_df)

        assert model.coef_.shape == (1, len(FEATURE_COLUMNS))

    def test_predict_win_probability_in_valid_range(self):
        df = make_games_df(200)
        train_df, test_df = chronological_split(df)
        model = train_model(train_df)

        prob = predict_win_probability(model, test_df.iloc[0])
        assert 0.0 <= prob <= 1.0


class TestWalkForwardSplits:
    def test_correct_number_of_folds(self):
        df = make_games_df(100)
        splits = walk_forward_splits(df, n_splits=4)
        assert len(splits) == 4

    def test_hand_verified_fold_boundaries(self):
        # 10 rows, n_splits=4 -> chunk_size = 10 // 5 = 2:
        # fold0: train[:2]  test[2:4]
        # fold1: train[:4]  test[4:6]
        # fold2: train[:6]  test[6:8]
        # fold3: train[:8]  test[8:10]  (last fold absorbs any remainder)
        df = make_games_df(10)
        splits = walk_forward_splits(df, n_splits=4)

        expected_train_sizes = [2, 4, 6, 8]
        expected_test_sizes = [2, 2, 2, 2]
        for (train_df, test_df), exp_train, exp_test in zip(splits, expected_train_sizes, expected_test_sizes):
            assert len(train_df) == exp_train
            assert len(test_df) == exp_test

    def test_folds_expand_and_never_leak_future_into_train(self):
        df = make_games_df(100)
        splits = walk_forward_splits(df, n_splits=4)

        for train_df, test_df in splits:
            assert train_df["date"].max() <= test_df["date"].min()

        # Training set must strictly grow fold over fold (expanding window).
        train_sizes = [len(train_df) for train_df, _ in splits]
        assert train_sizes == sorted(train_sizes)
        assert len(set(train_sizes)) == len(train_sizes)  # strictly increasing, no ties

    def test_test_folds_are_non_overlapping(self):
        df = make_games_df(100)
        splits = walk_forward_splits(df, n_splits=4)

        seen_dates = set()
        for _, test_df in splits:
            fold_dates = set(test_df["date"])
            assert not (fold_dates & seen_dates)
            seen_dates |= fold_dates

    def test_last_fold_absorbs_remainder_rows(self):
        # 11 rows doesn't divide evenly into 5 chunks (chunk_size=2, 1 leftover)
        # -- the leftover must land in the last fold's test set, not be dropped.
        df = make_games_df(11)
        splits = walk_forward_splits(df, n_splits=4)

        total_rows_covered = len(splits[0][0]) + sum(len(test_df) for _, test_df in splits)
        assert total_rows_covered == 11

    def test_invalid_n_splits_raises(self):
        df = make_games_df(10)
        with pytest.raises(ValueError):
            walk_forward_splits(df, n_splits=0)

    def test_too_few_rows_raises(self):
        df = make_games_df(3)
        with pytest.raises(ValueError):
            walk_forward_splits(df, n_splits=4)


class TestHasValueEdge:
    def test_flags_when_model_beats_market_by_threshold(self):
        assert has_value_edge(model_prob=0.58, market_no_vig_prob=0.52, threshold=0.03) is True

    def test_does_not_flag_below_threshold(self):
        assert has_value_edge(model_prob=0.53, market_no_vig_prob=0.52, threshold=0.03) is False

    def test_does_not_flag_when_market_favors_model(self):
        assert has_value_edge(model_prob=0.45, market_no_vig_prob=0.52, threshold=0.03) is False
