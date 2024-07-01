import hypothesis.strategies as st
from hypothesis import given


@given(st.integers())
def test_init_placeholder(number):
    assert number == number
