"""The feature-order contract.

This is the test that earns its place. A reordered feature list produces no
error anywhere: the model accepts the tensor, returns plausible numbers, and
every score is wrong.
"""
from serving.schemas import FEATURE_ORDER, Features


def test_vector_length_matches_the_declared_order():
    f = Features(amount=1, hour_of_day=0, merchant_risk=0.1, account_age_days=1,
                 txn_count_24h=0, distinct_countries_7d=1)
    assert len(f.to_vector()) == len(FEATURE_ORDER)


def test_vector_positions_match_the_declared_order():
    f = Features(amount=11.0, hour_of_day=22, merchant_risk=0.33, account_age_days=44,
                 txn_count_24h=5, distinct_countries_7d=6, is_cross_border=True)
    vector = dict(zip(FEATURE_ORDER, f.to_vector(), strict=True))
    assert vector["amount"] == 11.0
    assert vector["hour_of_day"] == 22.0
    assert vector["merchant_risk"] == 0.33
    assert vector["account_age_days"] == 44.0
    assert vector["is_cross_border"] == 1.0


def test_booleans_are_encoded_as_floats_not_left_as_bool():
    f = Features(amount=1, hour_of_day=0, merchant_risk=0, account_age_days=0,
                 txn_count_24h=0, distinct_countries_7d=0, is_cross_border=False)
    assert all(isinstance(v, float) for v in f.to_vector())
