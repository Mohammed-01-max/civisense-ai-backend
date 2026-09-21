import pytest
from datetime import datetime, timezone, timedelta
from ai.priority import calculate_priority, POLICY_VERSION
from models import Report
import math

class MockResult:
    def __init__(self, items):
        self.items = items
    def scalars(self):
        return self
    def all(self):
        return self.items

class MockSession:
    def __init__(self, mock_reports, current_time, current_category):
        self.mock_reports = mock_reports
        self.current_time = current_time
        self.current_category = current_category

    async def execute(self, query):
        filtered = []
        thirty_days_ago = self.current_time - timedelta(days=30)
        for r in self.mock_reports:
            if (r.created_at >= thirty_days_ago and
                r.created_at < self.current_time and
                r.triage_category == self.current_category):
                filtered.append(r)
        return MockResult(filtered)

def get_point_at_distance(lat, lon, distance_m):
    # approximate using 1 deg lat = 111,320 meters
    # this creates a simple point exactly north
    lat_offset = distance_m / 111320.0
    return lat + lat_offset, lon

@pytest.mark.asyncio
async def test_priority_tc1():
    now = datetime.now(timezone.utc)
    db = MockSession([], now, "pothole")
    result = await calculate_priority(db, "HIGH", "TRUE", "pothole", 17.0, 78.0, now)
    assert result["level"] == "HIGH"
    assert result["reason_code"] == "PRI_SEV_EVID"
    assert result["factors"]["possible_recurrence"] == "FALSE"

@pytest.mark.asyncio
async def test_priority_tc2():
    now = datetime.now(timezone.utc)
    db = MockSession([], now, "garbage")
    result = await calculate_priority(db, "HIGH", "UNKNOWN", "garbage", 17.0, 78.0, now)
    assert result["level"] == "MEDIUM"
    assert result["reason_code"] == "PRI_SEV_NO_EVID"

@pytest.mark.asyncio
async def test_priority_tc3():
    now = datetime.now(timezone.utc)
    hist = Report(latitude=17.0, longitude=78.0, created_at=now - timedelta(days=1), triage_category="street_light")
    db = MockSession([hist], now, "street_light")
    result = await calculate_priority(db, None, "TRUE", "street_light", 17.0, 78.0, now)
    assert result["level"] == "HIGH"
    assert result["reason_code"] == "PRI_REC_EVID"
    assert result["factors"]["possible_recurrence"] == "TRUE"

@pytest.mark.asyncio
async def test_priority_tc4():
    now = datetime.now(timezone.utc)
    db = MockSession([], now, None)
    result = await calculate_priority(db, None, "FALSE", None, 17.0, 78.0, now)
    assert result["level"] == "LOW"
    assert result["reason_code"] == "PRI_ROUTINE"
    assert result["factors"]["possible_recurrence"] == "UNKNOWN"

@pytest.mark.asyncio
async def test_priority_tc5_exception():
    class ErrorSession:
        async def execute(self, q):
            raise Exception("DB error")
    now = datetime.now(timezone.utc)
    with pytest.raises(Exception):
        await calculate_priority(ErrorSession(), "HIGH", "TRUE", "cat", 17.0, 78.0, now)

@pytest.mark.asyncio
async def test_priority_category_mismatch():
    now = datetime.now(timezone.utc)
    hist = Report(latitude=17.0, longitude=78.0, created_at=now - timedelta(days=1), triage_category="garbage")
    db = MockSession([hist], now, "pothole")
    result = await calculate_priority(db, "LOW", "TRUE", "pothole", 17.0, 78.0, now)
    assert result["factors"]["possible_recurrence"] == "FALSE"

@pytest.mark.asyncio
async def test_priority_haversine_exact_50m():
    from ai.priority import haversine
    now = datetime.now(timezone.utc)
    # 50.0 meters is approx 0.0004496605 degrees.
    lon2 = 0.0004496605206422906
    dist = haversine(0.0, 0.0, 0.0, lon2)
    assert dist <= 50.0
    assert dist > 49.99

    hist = Report(latitude=0.0, longitude=lon2, created_at=now - timedelta(days=1), triage_category="pothole")
    db = MockSession([hist], now, "pothole")
    result = await calculate_priority(db, "LOW", "TRUE", "pothole", 0.0, 0.0, now)
    assert result["factors"]["possible_recurrence"] == "TRUE"

@pytest.mark.asyncio
async def test_priority_haversine_just_over_50m():
    from ai.priority import haversine
    now = datetime.now(timezone.utc)
    # 50.01 meters
    lon2 = 0.0004497504866943548
    assert haversine(0.0, 0.0, 0.0, lon2) > 50.0

    hist = Report(latitude=0.0, longitude=lon2, created_at=now - timedelta(days=1), triage_category="pothole")
    db = MockSession([hist], now, "pothole")
    result = await calculate_priority(db, "LOW", "TRUE", "pothole", 0.0, 0.0, now)
    assert result["factors"]["possible_recurrence"] == "FALSE"

@pytest.mark.asyncio
async def test_priority_exact_30_days_boundary():
    now = datetime.now(timezone.utc)
    hist = Report(latitude=17.0, longitude=78.0, created_at=now - timedelta(days=30), triage_category="pothole")
    db = MockSession([hist], now, "pothole")
    result = await calculate_priority(db, "LOW", "TRUE", "pothole", 17.0, 78.0, now)
    assert result["factors"]["possible_recurrence"] == "TRUE"

@pytest.mark.asyncio
async def test_priority_just_over_30_days():
    now = datetime.now(timezone.utc)
    hist = Report(latitude=17.0, longitude=78.0, created_at=now - timedelta(days=30, seconds=1), triage_category="pothole")
    db = MockSession([hist], now, "pothole")
    result = await calculate_priority(db, "LOW", "TRUE", "pothole", 17.0, 78.0, now)
    assert result["factors"]["possible_recurrence"] == "FALSE"

@pytest.mark.asyncio
async def test_priority_missing_historical_category():
    now = datetime.now(timezone.utc)
    hist = Report(latitude=17.0, longitude=78.0, created_at=now - timedelta(days=1), triage_category=None)
    db = MockSession([hist], now, "pothole")
    result = await calculate_priority(db, "LOW", "TRUE", "pothole", 17.0, 78.0, now)
    assert result["factors"]["possible_recurrence"] == "FALSE"

@pytest.mark.asyncio
async def test_priority_missing_historical_coordinates():
    now = datetime.now(timezone.utc)
    hist = Report(latitude=None, longitude=None, created_at=now - timedelta(days=1), triage_category="pothole")
    db = MockSession([hist], now, "pothole")
    result = await calculate_priority(db, "LOW", "TRUE", "pothole", 17.0, 78.0, now)
    assert result["factors"]["possible_recurrence"] == "FALSE"

@pytest.mark.asyncio
async def test_priority_future_report_exclusion():
    now = datetime.now(timezone.utc)
    hist = Report(latitude=17.0, longitude=78.0, created_at=now + timedelta(days=1), triage_category="pothole")
    db = MockSession([hist], now, "pothole")
    result = await calculate_priority(db, "LOW", "TRUE", "pothole", 17.0, 78.0, now)
    assert result["factors"]["possible_recurrence"] == "FALSE"

@pytest.mark.asyncio
async def test_priority_multiple_candidates():
    now = datetime.now(timezone.utc)
    hist1 = Report(latitude=17.0, longitude=78.0, created_at=now - timedelta(days=40), triage_category="pothole") # fails time
    hist2 = Report(latitude=18.0, longitude=78.0, created_at=now - timedelta(days=1), triage_category="pothole")  # fails dist
    hist3 = Report(latitude=17.0, longitude=78.0, created_at=now - timedelta(days=2), triage_category="garbage")  # fails cat
    hist4 = Report(latitude=17.0, longitude=78.0, created_at=now - timedelta(days=2), triage_category="pothole")  # PASS

    db = MockSession([hist1, hist2, hist3, hist4], now, "pothole")
    result = await calculate_priority(db, "LOW", "TRUE", "pothole", 17.0, 78.0, now)
    assert result["factors"]["possible_recurrence"] == "TRUE"

@pytest.mark.asyncio
async def test_priority_unknown_never_silently_converted():
    now = datetime.now(timezone.utc)
    result = await calculate_priority(MockSession([], now, "cat"), "HIGH", "UNKNOWN", "cat", 17.0, 78.0, now)
    assert result["factors"]["yolo_evidence_state"] == "UNKNOWN"
