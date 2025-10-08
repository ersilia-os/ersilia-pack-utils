import csv
import json
import struct
import numpy as np


Smiles=list[str]
Header=list[str]

def read_smiles_csv(in_file:str)->tuple[Header,Smiles]:
    """Read SMILES strings from a CSV file."""

    with open(in_file, "r") as f:
        reader = csv.reader(f)
        cols = next(reader)
        data = [r[0] for r in reader]
        return cols, data


def read_smiles_bin(in_file:str)->tuple[Header,Smiles]:
    """Read SMILES strings from a custom binary file."""

    with open(in_file, "rb") as f:
        data = f.read()

        mv = memoryview(data)
        nl = mv.tobytes().find(b"\n")
        meta = json.loads(mv[:nl].tobytes().decode("utf-8"))
        cols = meta.get("columns", [])
        count = meta.get("count", 0)

        smiles_list = [None] * count
        offset = nl + 1
        for i in range(count):
            (length,) = struct.unpack_from(">I", mv, offset)
            offset += 4
            smiles_list[i] = mv[offset : offset + length].tobytes().decode("utf-8")
            offset += length

        return cols, smiles_list


def read_smiles(in_file:str)->tuple[Header,Smiles]:
    """
    Dispatch to the appropriate SMILES reader based on file suffix.

    If the path ends with `.bin`, uses :func:`read_smiles_bin`; otherwise uses
    :func:`read_smiles_csv`
    """
    
    if in_file.endswith(".bin"):
        return read_smiles_bin(in_file)
    return read_smiles_csv(in_file)


def write_out_csv(results, header, file:str)->None:
    """
    Write rows to a CSV file.
    Writes `header` as the first row followed by each row in `results`
    """

    with open(file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in results:
            writer.writerow(r)


def write_out_bin(results, header, file:str)->None:
    """
    Write a 2D array of int32s to a custom binary file
    """

    arr = np.asarray(results, dtype=np.int32)
    meta = {"columns": header, "shape": arr.shape, "dtype": "int32"}
    meta_bytes = (json.dumps(meta) + "\n").encode("utf-8")

    with open(file, "wb") as f:
        f.write(meta_bytes)
        f.truncate(len(meta_bytes) + arr.nbytes)

    m = np.memmap(
    file, dtype=arr.dtype, mode="r+", offset=len(meta_bytes), shape=arr.shape
    )
    m[:] = arr
    m.flush()

def write_out(results, header, file:str)->None:
    """
    Dispatch to CSV or binary writer based on file suffix.

    If `file` ends with `.bin`, calls :func:`write_out_bin`. If it ends with `.csv`,
    calls :func:`write_out_csv`. Otherwise, raises a `ValueError`.
    """

    if file.endswith(".bin"):
        write_out_bin(results, header, file)
    elif file.endswith(".csv"):
        write_out_csv(results, header, file)
    else:
        raise ValueError(f"Unsupported extension for {file!r}")