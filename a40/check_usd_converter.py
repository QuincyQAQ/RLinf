#!/usr/bin/env python3
"""Test actual USD -> URDF conversion without Kit, simulation, or GPU work."""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from nvidia.srl.from_usd.to_urdf import UsdToUrdf
    assert sys.prefix != sys.base_prefix
    assert all("pip_prebundle" not in p and ".asset_tools" not in p for p in sys.path)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True, timeout=60)
    with tempfile.TemporaryDirectory(prefix="usd_converter_cpu_") as temporary:
        folder = Path(temporary)
        source, target = folder / "two_links.usda", folder / "two_links.urdf"
        stage = Usd.Stage.CreateNew(str(source))
        robot = UsdGeom.Xform.Define(stage, "/Robot")
        stage.SetDefaultPrim(robot.GetPrim())
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
        for name in ("base", "leg"):
            body = UsdGeom.Xform.Define(stage, "/Robot/" + name).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            mass = UsdPhysics.MassAPI.Apply(body)
            mass.CreateMassAttr(1.0)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.1, .1, .1))
            cube = UsdGeom.Cube.Define(stage, "/Robot/" + name + "/shape")
            cube.CreateSizeAttr(.1)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/hinge")
        joint.CreateBody0Rel().SetTargets(["/Robot/base"])
        joint.CreateBody1Rel().SetTargets(["/Robot/leg"])
        joint.CreateAxisAttr("Z")
        joint.CreateLowerLimitAttr(-45)
        joint.CreateUpperLimitAttr(45)
        stage.GetRootLayer().Save()
        converter = UsdToUrdf.init_from_file(str(source))
        converter.save_to_file(urdf_output_path=str(target),
                               mesh_dir=str(folder / "meshes"),
                               visualize_collision_meshes=False)
        tree = ET.parse(target).getroot()
        links, joints = tree.findall("link"), tree.findall("joint")
        assert len(links) >= 2 and any(j.get("type") == "revolute" for j in joints)
        for j in joints:
            assert j.find("parent").get("link") in {link.get("name") for link in links}
            assert j.find("child").get("link") in {link.get("name") for link in links}
        conversion = {"links": len(links), "joints": len(joints), "urdf_xml_parsed": True}
    packages = ["nvidia-srl-usd-to-urdf", "nvidia-srl-usd", "nvidia-srl-base", "usd-core", "lxml", "numpy"]
    report = {"status": "passed", "finished_unix": time.time(),
              "versions": {p: metadata.version(p) for p in packages},
              "python_prefix": sys.prefix, "usd_origin": Usd.__file__,
              "pip_check_clean": True, "conversion": conversion,
              "simulation_started": False, "gpu_used": False}
    (ROOT / "validation/usd_converter_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
