import zipfile

from tools.export_trade_snapshot import _member


def test_member_prefers_canonical_root_csv(tmp_path):
    archive_name = "BTCUSDT-aggTrades-2021-12.zip"
    canonical = "BTCUSDT-aggTrades-2021-12.csv"
    duplicate = (
        "fsx-data/collector_data/data/spot/monthly/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2021-12.csv"
    )

    path = tmp_path / archive_name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(canonical, "1,2,3\n")
        zf.writestr(duplicate, "1,2,3\n")

    with zipfile.ZipFile(path) as zf:
        assert _member(zf, archive_name) == canonical
