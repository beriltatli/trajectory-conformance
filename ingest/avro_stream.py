import tarfile
from collections.abc import Iterator
from pathlib import Path

import fastavro
import numpy as np
import pandas as pd
import pyarrow as pa

STATE_COLUMNS = (
    "time", "icao24", "lat", "lon", "velocity", "heading", "vertrate", "callsign", "onground",
    "alert", "spi", "squawk", "baroaltitude", "geoaltitude", "lastposupdate", "lastcontact",
)
FLOAT_COLUMNS = (
    "lat", "lon", "velocity", "heading", "vertrate", "baroaltitude", "geoaltitude",
    "lastposupdate", "lastcontact",
)

STATE_ARROW_SCHEMA = pa.schema([
    ("time", pa.int64()), ("icao24", pa.string()), ("lat", pa.float64()), ("lon", pa.float64()),
    ("velocity", pa.float64()), ("heading", pa.float64()), ("vertrate", pa.float64()),
    ("callsign", pa.string()), ("onground", pa.bool_()), ("alert", pa.bool_()), ("spi", pa.bool_()),
    ("squawk", pa.string()), ("baroaltitude", pa.float64()), ("geoaltitude", pa.float64()),
    ("lastposupdate", pa.float64()), ("lastcontact", pa.float64()),
])
MEASUREMENT_ARROW_SCHEMA = STATE_ARROW_SCHEMA.append(pa.field("contact_before", pa.float64())).append(pa.field("flight_id", pa.string()))

# Writer schema of the OpenSky state-vector samples (read from states_2022-06-20-14.avro);
# kept here so tests can build synthetic files the reader cannot tell apart from real ones.
STATE_SCHEMA = {
    "type": "record",
    "name": "org.opensky.avro.export.StateVector",
    "fields": [
        {"name": "alert", "type": "boolean"},
        {"name": "baroaltitude", "type": ["null", "double"]},
        {"name": "callsign", "type": ["null", "string"]},
        {"name": "geoaltitude", "type": ["null", "double"]},
        {"name": "heading", "type": ["null", "double"]},
        {"name": "icao24", "type": "string"},
        {"name": "lastcontact", "type": ["null", "double"]},
        {"name": "lastposupdate", "type": ["null", "double"]},
        {"name": "lat", "type": ["null", "double"]},
        {"name": "lon", "type": ["null", "double"]},
        {"name": "onground", "type": "boolean"},
        {"name": "spi", "type": "boolean"},
        {"name": "squawk", "type": ["null", "string"]},
        {"name": "time", "type": "int"},
        {"name": "velocity", "type": ["null", "double"]},
        {"name": "vertrate", "type": ["null", "double"]},
    ],
}


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records, columns=STATE_COLUMNS)
    for col in FLOAT_COLUMNS:
        frame[col] = frame[col].astype("float64")
    frame["time"] = frame["time"].astype("int64")
    frame["onground"] = frame["onground"].astype(bool)
    # Callsigns are space-padded to 8 characters in the source.
    frame["callsign"] = frame["callsign"].str.strip().replace("", np.nan)
    return frame


def iter_state_chunks(tar_path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    # "r|" reads the tar as a stream: the avro member is decoded block by block and never
    # extracted, so memory is bounded by chunk_rows, not by the hour.
    with tarfile.open(tar_path, mode="r|") as archive:
        for member in archive:
            if not member.name.endswith(".avro"):
                continue
            handle = archive.extractfile(member)
            buffer: list[dict] = []
            for record in fastavro.reader(handle):
                buffer.append(record)
                if len(buffer) >= chunk_rows:
                    yield records_to_frame(buffer)
                    buffer = []
            if buffer:
                yield records_to_frame(buffer)
