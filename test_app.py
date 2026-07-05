import pytest
from streamlit.testing.v1 import AppTest

def check_markdown(at, text):
    """Helper to check if text exists in any markdown block."""
    return any(text in str(md.value) for md in at.markdown)

def test_1_app_initialization():
    """1. Verify the app loads successfully without exceptions."""
    at = AppTest.from_file("app.py").run(timeout=15)
    assert not at.exception

def test_2_title_and_layout():
    """2. Check that the main title renders correctly."""
    at = AppTest.from_file("app.py").run(timeout=15)
    assert at.title[0].value == "🛒 Shopper Spectrum Dashboard"
    assert check_markdown(at, "Discover Customer Segments & Personalized Product Recommendations")

def test_3_tab_navigation():
    """3. Verify that both tabs exist."""
    at = AppTest.from_file("app.py").run(timeout=15)
    assert len(at.tabs) == 2

def test_4_empty_product_selection():
    """4. Simulate clicking 'Get Recommendations' without selecting a product."""
    at = AppTest.from_file("app.py").run(timeout=15)
    at.button[0].click().run(timeout=15)
    assert at.warning[0].value == "Please select a product from the dropdown."

def test_5_valid_product_recommendation():
    """5. Simulate selecting a valid product and ensure recommendations generate."""
    at = AppTest.from_file("app.py").run(timeout=15)
    # Select the second item (index 1) which is a valid product
    at.selectbox[0].set_value(at.selectbox[0].options[1]).run(timeout=15)
    at.button[0].click().run(timeout=15)
    assert at.subheader[0].value == "Top 5 Recommended Products:"

def test_6_ui_tab_2_layout():
    """6. Verify the UI layout of the Customer Segmentation tab."""
    at = AppTest.from_file("app.py").run(timeout=15)
    assert at.header[1].value == "Predict Customer Segment"

def test_7_default_profile_prediction():
    """7. Simulate clicking 'Predict Cluster' using default values."""
    at = AppTest.from_file("app.py").run(timeout=15)
    at.button[1].click().run(timeout=15)
    assert not at.exception
    assert check_markdown(at, "Predicted Customer Segment")

def test_8_high_value_profile():
    """8. Test Recency=1, Frequency=50, Monetary=5000 (High-Value)."""
    at = AppTest.from_file("app.py").run(timeout=15)
    at.number_input[0].set_value(1)
    at.number_input[1].set_value(50)
    at.number_input[2].set_value(5000.0)
    at.button[1].click().run(timeout=15)
    assert not at.exception
    assert check_markdown(at, "High-Value")

def test_9_at_risk_profile():
    """9. Test Recency=300, Frequency=1, Monetary=10 (At-Risk)."""
    at = AppTest.from_file("app.py").run(timeout=15)
    at.number_input[0].set_value(300)
    at.number_input[1].set_value(1)
    at.number_input[2].set_value(10.0)
    at.button[1].click().run(timeout=15)
    assert not at.exception
    assert check_markdown(at, "At-Risk")

def test_10_extreme_outlier_stress_test():
    """10. Test crazy outliers (Recency=10000, Frequency=500, Monetary=60) to ensure no crashes."""
    at = AppTest.from_file("app.py").run(timeout=15)
    at.number_input[0].set_value(10000)
    at.number_input[1].set_value(500)
    at.number_input[2].set_value(60.0)
    at.button[1].click().run(timeout=15)
    assert not at.exception
    assert check_markdown(at, "Regular")
