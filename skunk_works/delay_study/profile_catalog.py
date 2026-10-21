"""Explicit, hash-checked selection of preserved or audited refined profiles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProfileCatalog:
    def __init__(self,results,selection=None):
        self.root = Path(results).resolve()
        requested = Path(selection) if selection is not None else Path("profile_index.json")
        self.path = (requested if requested.is_absolute() else self.root/requested).resolve()
        self.inputs = {}
        self.data = self.read(self.path)
        self.entries = self.data["profiles"]
        if len(self.entries) != 300:
            raise RuntimeError("The study catalog requires all 300 profile selections")
        refined = {n for n,e in self.entries.items() if e.get("refined",False)}
        marked = any(k in self.data for k in ("refinement_manifest_sha256","refinement_audit_sha256"))
        self.kind = "audited_selected_refinement" if marked or refined else "preserved_first_pass"
        if self.path.name == "selected_profile_index.json" and self.kind != "audited_selected_refinement":
            raise RuntimeError("A selected profile catalog needs audited refinement provenance")
        if self.kind == "audited_selected_refinement":
            self.check_refinement(refined)
        self.selection = {"path":str(self.path),"sha256":sha(self.path),"kind":self.kind,
                          "profile_count":len(self.entries),"refined_profiles":len(refined)}

    def read(self,path,expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise RuntimeError("Profile selection receipt bytes changed: "+str(path))
        self.inputs[str(path)] = digest
        return json.loads(path.read_text())

    def check_refinement(self,refined):
        if len(refined) != 18 or not all(k in self.data for k in ("refinement_manifest_sha256","refinement_audit_sha256")):
            raise RuntimeError("Selected catalog requires all 18 audited refinements")
        audit_path = self.root/self.data.get("refinement_audit_path","profile_refinement_audit.json")
        manifest_path = self.root/self.data.get("refinement_manifest_path","profile_refinement/manifest.json")
        if not audit_path.resolve().is_relative_to(self.root) or not manifest_path.resolve().is_relative_to(self.root):
            raise RuntimeError("Refinement receipts must belong to this result collection")
        audit = self.read(audit_path,self.data["refinement_audit_sha256"])
        manifest = self.read(manifest_path,self.data["refinement_manifest_sha256"])
        audited = {r["profile"] for r in audit["profiles"]}
        if audit["status"] != "passed" or audit["refined_profiles"] != 18 or len(audit["profiles"]) != 18 or audited != refined:
            raise RuntimeError("Refinement audit does not cover the selected profiles")
        if manifest["status"] != "complete" or manifest["failed"] or set(manifest["completed"]) != refined:
            raise RuntimeError("Selected refinement is not a complete terminal phase")
        first_path = self.root/"profile_index.json"
        first = self.read(first_path,manifest["identity"]["source_catalog_sha256"])["profiles"]
        if set(first) != set(self.entries):
            raise RuntimeError("Selected catalog lost or added a first-pass profile")
        for name,entry in self.entries.items():
            if name in refined:
                if entry.get("source_first_pass_path") != first[name]["path"] or entry.get("analytic_family_inclusion_failed",False):
                    raise RuntimeError("Refinement lacks its original selection or still has an inclusion flag")
            elif entry != first[name]:
                raise RuntimeError("An unrefined selection changed silently: "+name)

    def profile_path(self,name):
        entry = self.entries[name]
        if entry.get("analytic_family_inclusion_failed",False):
            raise RuntimeError("Coarse profile excludes exact equivalent parameters; use an audited refinement: "+name)
        path = (self.root/entry["path"]).resolve()
        if not path.is_relative_to(self.root) or sha(path) != entry["sha256"]:
            raise RuntimeError("Selected profile file path/bytes changed: "+name)
        self.inputs[str(path)] = entry["sha256"]
        return path

    def verify_manifest(self,manifest):
        saved = manifest.get("profile_catalog")
        if saved is not None:
            if any(saved.get(k) != self.selection[k] for k in ("sha256","kind","profile_count","refined_profiles")):
                raise RuntimeError("Artifact used another explicit profile catalog")
        else:
            # Earlier artifacts predate explicit catalog metadata. Their
            # preserved input hashes identify only the first-pass selection.
            matching = [h for p,h in manifest["input_sha256"].items() if Path(p).name == self.path.name]
            if self.kind != "preserved_first_pass" or matching != [self.selection["sha256"]]:
                raise RuntimeError("Legacy artifact does not identify this profile catalog")
