#!/usr/bin/env python3
"""Mirror original Go2/ground/sky assets over HTTPS; USD parsing is CPU-only.

An isolated .asset_tools USD wheel is used only by this downloader. It is never
added to the simulation Python entrypoint, which retains Isaac Sim's USD stack.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
PREFIX = "Assets/Isaac/5.1/"
MIRROR = ROOT / "assets/isaac_sim_5.1"
FOLDERS = ["Isaac/IsaacLab/Robots/Unitree/Go2/",
           "Isaac/IsaacLab/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"]
SKY = "Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
CACHE_PATH = ROOT / "validation/asset_download_cache.json"
CACHE = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
BUILTIN_MDL = {}


def usd_asset_references(text):
    # Empty USD asset values are @@. Match them too, then discard, so their
    # delimiters cannot consume adjacent metadata as a spurious asset path.
    return [value for value in re.findall(r"@([^@]*)@",text) if value]


def location(key):
    path = MIRROR / key.removeprefix(PREFIX)
    if not path.resolve().is_relative_to(MIRROR.resolve()):
        raise ValueError("Unsafe asset path")
    return path


def list_keys(folder):
    token = None
    keys = []
    while True:
        params = {"list-type": "2", "prefix": PREFIX + folder}
        if token:
            params["continuation-token"] = token
        with urllib.request.urlopen(BUCKET + "/?" + urllib.parse.urlencode(params), timeout=30) as response:
            xml = ET.fromstring(response.read())
        keys += [node.find("s3:Key", NS).text for node in xml.findall("s3:Contents", NS)
                 if not node.find("s3:Key", NS).text.endswith("/")
                 and not any(part.startswith(".") for part in Path(node.find("s3:Key", NS).text).parts)]
        if xml.findtext("s3:IsTruncated", namespaces=NS) != "true":
            return keys
        token = xml.findtext("s3:NextContinuationToken", namespaces=NS)


def download(key):
    path = location(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = BUCKET + "/" + urllib.parse.quote(key, safe="/")
    cached = CACHE.get(key)
    if cached and path.is_file() and path.stat().st_size == cached["bytes"]:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
            if response.headers.get("ETag") == cached.get("etag"):
                return dict(cached, reused=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                length = response.headers.get("Content-Length")
                temporary = path.with_name(path.name + ".download")
                with temporary.open("wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                if length is not None and temporary.stat().st_size != int(length):
                    raise RuntimeError("Incomplete asset download")
                temporary.replace(path)
                return {"key": key, "url": url, "local_path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size, "etag": response.headers.get("ETag")}
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError("Asset download failed: " + url + ": " + str(exc)) from exc
            time.sleep(2)


def reference_key(parent_key, reference):
    reference = reference.strip().replace("\\", "/")
    if not reference or reference.startswith(("anon:", "::")):
        return None
    if "/" not in reference and reference.endswith(".mdl"):
        for base in [Path("/isaac-sim/kit/mdl"), Path("/isaac-sim/mdl")]:
            if base.exists():
                match = next(base.rglob(reference), None)
                if match is not None:
                    BUILTIN_MDL[reference] = str(match)
                    return None
    url = urllib.parse.urljoin(BUCKET + "/" + parent_key, reference)
    parsed = urllib.parse.urlsplit(url)
    accepted = {"omniverse-content-production.s3-us-west-2.amazonaws.com",
                "omniverse-content-production.s3.us-west-2.amazonaws.com"}
    if parsed.hostname not in accepted or not parsed.path.lstrip("/").startswith(PREFIX):
        raise ValueError("Asset reference outside selected official 5.1 tree: " + url)
    return urllib.parse.unquote(parsed.path.lstrip("/"))


def main():
    # Tools are deliberately absent from container_python.sh / the training import path.
    if os.environ.get("GO2_ASSET_TOOLS_BOOTSTRAPPED") != "1":
        environment = dict(os.environ, GO2_ASSET_TOOLS_BOOTSTRAPPED="1")
        environment["LD_LIBRARY_PATH"] = str(ROOT / ".asset_tools/usd_core.libs") + ":" + environment.get("LD_LIBRARY_PATH", "")
        environment["PYTHONPATH"] = str(ROOT / ".asset_tools") + ":" + environment.get("PYTHONPATH", "")
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())], environment)
    sys.path.insert(0, str(ROOT / ".asset_tools"))
    from pxr import Sdf, Usd
    MIRROR.mkdir(parents=True, exist_ok=True)
    pending = {PREFIX + SKY, PREFIX + 'Isaac/Environments/Grid/default_environment.usd',
               PREFIX + 'Isaac/Props/UIElements/arrow_x.usd',
               PREFIX + 'Isaac/Props/UIElements/frame_prim.usd'}
    for folder in FOLDERS:
        pending.update(list_keys(folder))
    records = {}
    references = []
    total = 0
    while pending:
        if len(records) + len(pending) > 500 or total > 2 * 1024**3:
            raise RuntimeError("Asset mirror exceeded the bounded Go2 preparation scope")
        batch = sorted(pending)
        pending = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for record in executor.map(download, batch):
                records[record["key"]] = record
                CACHE[record["key"]] = record
                cache_tmp = CACHE_PATH.with_suffix(".tmp")
                cache_tmp.write_text(json.dumps(CACHE, indent=2))
                cache_tmp.replace(CACHE_PATH)
                total += record["bytes"]
                print("DOWNLOADED", record["key"], record["bytes"], flush=True)
        for key in batch:
            path = location(key)
            if path.suffix.lower() in (".usd", ".usda", ".usdc"):
                layer = Sdf.Layer.FindOrOpen(str(path))
                if not layer:
                    raise RuntimeError("USD layer cannot be parsed: " + str(path))
                text = layer.ExportToString()
                found = usd_asset_references(text)
            elif path.suffix.lower() == ".mdl":
                text = path.read_text(errors="replace")
                found = re.findall(r'"([^"\n]+\.(?:png|jpe?g|tiff?|exr|hdr|mdl))"', text, re.I)
            else:
                continue
            for reference in sorted(set(found)):
                dependency = reference_key(key, reference)
                if dependency is None:
                    continue
                references.append({"parent": key, "authored_reference": reference, "key": dependency,
                                   "absolute_url_in_source": bool(urllib.parse.urlsplit(reference).scheme)})
                if dependency not in records:
                    pending.add(dependency)
    # Check that all listed dependencies have local files; do not launch Usd physics or Kit.
    missing = [item for item in references if not location(item["key"]).is_file()]
    stage = Usd.Stage.Open(str(location(PREFIX + "Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd")))
    joints = [str(prim.GetPath()) for prim in stage.Traverse() if prim.GetTypeName() == "PhysicsRevoluteJoint"]
    if len(joints) != 12:
        raise RuntimeError("Expected 12 revolute joints in the original Go2 USD, found " + str(len(joints)))
    report = {"status": "passed" if not missing else "failed", "finished_unix": time.time(),
              "official_cloud_root": BUCKET + "/" + PREFIX.rstrip("/"),
              "local_cloud_root": str(MIRROR), "file_count": len(records), "total_bytes": total,
              "files": list(records.values()), "asset_references": references, "missing": missing,
              "builtin_mdl": BUILTIN_MDL, "robot_revolute_joint_count": len(joints), "robot_joint_paths": joints,
              "asset_contents_rewritten": False, "training_asset_root_overridden": False,
              "usd_tool_version": list(Usd.GetVersion()), "simulation_started": False, "gpu_used": False,
              "runtime_shader_and_physics_validation": "pending A40 simulation checks"}
    (ROOT / "validation/asset_mirror.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("files", "asset_references")}, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
