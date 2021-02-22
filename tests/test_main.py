def test_package_importable():
    from book_rec.main import main

    assert main() == "hello world"
