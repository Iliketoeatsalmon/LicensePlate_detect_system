from lpr.plate import assemble_plate_and_province, group_by_line
from lpr.thai import get_thai_character, split_license_plate_and_province


def tok(x, cls, y=10):
    return (x, cls, (x, y, x + 5, y + 10))


def test_get_thai_character_maps_and_falls_back():
    assert get_thai_character("A01") == "ก"
    assert get_thai_character("5") == "5"


def test_split_plate_and_province():
    assert split_license_plate_and_province("1กข234กรุงเทพ") == ("1กข234", "กรุงเทพ")


def test_single_row_car_plate():
    tokens = [tok(30, "2"), tok(10, "A01"), tok(20, "A02"), tok(40, "BKK")]
    lines = group_by_line(tokens)
    assert len(lines) == 1
    plate, province = assemble_plate_and_province(lines)
    assert plate == "กข2"
    assert province == "กรุงเทพมหานคร"


def test_two_rows_split_by_y():
    tokens = [tok(10, "A01", y=10), tok(20, "5", y=10), tok(10, "BKK", y=100)]
    lines = group_by_line(tokens, y_tol=30)
    assert len(lines) == 2
    plate, province = assemble_plate_and_province(lines)
    assert plate == "ก5"
    assert province == "กรุงเทพมหานคร"


def test_empty():
    assert assemble_plate_and_province([]) == ("", "")
